//! Sharded CSV writer for the event stream + node tables.
//!
//! Output layout (out_dir/):
//!   edges_00000.csv, edges_00001.csv, …   sharded by target size
//!   nodes.csv                              one row per node
//! Headers are written per shard; numeric attrs render empty when NaN
//! (not applicable for that event type).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use agora_core::{EventBatch, EventSink, NodeBatch};

pub struct CsvSink {
    dir: PathBuf,
    shard_bytes_target: u64,
    event_type_names: Vec<String>,
    intent_names: Vec<String>,
    /// attr name -> category values (decode codes to strings on write).
    attr_dicts: Vec<(String, Vec<String>)>,
    /// Per union-column dictionary, resolved on first batch.
    col_dicts: Option<Vec<Option<Vec<String>>>>,
    shard_idx: u32,
    bytes_in_shard: u64,
    writer: Option<BufWriter<File>>,
    header: Option<String>,
    pub shards_written: Vec<PathBuf>,
}

impl CsvSink {
    pub fn new(
        dir: PathBuf,
        shard_size_mb: u64,
        event_type_names: Vec<String>,
        intent_names: Vec<String>,
        attr_dicts: Vec<(String, Vec<String>)>,
    ) -> anyhow::Result<CsvSink> {
        std::fs::create_dir_all(&dir)?;
        Ok(CsvSink {
            dir,
            shard_bytes_target: shard_size_mb * 1024 * 1024,
            event_type_names,
            intent_names,
            attr_dicts,
            col_dicts: None,
            shard_idx: 0,
            bytes_in_shard: 0,
            writer: None,
            header: None,
            shards_written: Vec::new(),
        })
    }

    fn roll_shard(&mut self) -> anyhow::Result<&mut BufWriter<File>> {
        if self.writer.is_none() || self.bytes_in_shard >= self.shard_bytes_target {
            if let Some(mut w) = self.writer.take() {
                w.flush()?;
            }
            let path = self.dir.join(format!("edges_{:05}.csv", self.shard_idx));
            self.shard_idx += 1;
            self.bytes_in_shard = 0;
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
            if let Some(h) = &self.header {
                w.write_all(h.as_bytes())?;
            }
            self.shards_written.push(path);
            self.writer = Some(w);
        }
        Ok(self.writer.as_mut().expect("just ensured"))
    }
}

impl EventSink for CsvSink {
    fn write_batch(&mut self, batch: &EventBatch) -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        if self.header.is_none() {
            let mut h = String::from("src,dst,t,event_type,label,anomaly_id");
            for a in &batch.attr_names {
                h.push(',');
                h.push_str(a);
            }
            h.push('\n');
            self.header = Some(h);
        }
        // Resolve per-column dictionaries against the union schema once.
        if self.col_dicts.is_none() {
            self.col_dicts = Some(
                batch
                    .attr_names
                    .iter()
                    .map(|name| {
                        self.attr_dicts
                            .iter()
                            .find(|(n, _)| n == name)
                            .map(|(_, vals)| vals.clone())
                    })
                    .collect(),
            );
        }
        let col_dicts = self.col_dicts.clone().expect("just set");
        // Render rows into a reusable buffer, then write once.
        let mut buf = String::with_capacity(batch.len() * 48);
        for i in 0..batch.len() {
            use std::fmt::Write as _;
            let et = &self.event_type_names[batch.event_type[i] as usize];
            let label = &self.intent_names[batch.label[i] as usize];
            write!(
                buf,
                "{},{},{},{et},{label},{}",
                batch.src[i], batch.dst[i], batch.t[i], batch.anomaly_id[i]
            )
            .unwrap();
            for (c, col) in batch.attrs.iter().enumerate() {
                let v = col[i];
                if v.is_nan() {
                    buf.push(',');
                } else if let Some(dict) = &col_dicts[c] {
                    let code = (v as usize).min(dict.len().saturating_sub(1));
                    write!(buf, ",{}", dict[code]).unwrap();
                } else if v == v.trunc() && v.abs() < 1e15 {
                    write!(buf, ",{}", v as i64).unwrap();
                } else {
                    write!(buf, ",{v:.4}").unwrap();
                }
            }
            buf.push('\n');
        }
        let w = self.roll_shard()?;
        w.write_all(buf.as_bytes())?;
        self.bytes_in_shard += buf.len() as u64;
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        if let Some(mut w) = self.writer.take() {
            w.flush()?;
        }
        Ok(())
    }

    fn write_nodes(&mut self, batch: &NodeBatch) -> anyhow::Result<()> {
        let path = self.dir.join("nodes.csv");
        let exists = path.exists();
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        let mut w = BufWriter::with_capacity(1 << 20, file);
        if !exists {
            let mut h = String::from("id,entity_type");
            for a in &batch.attr_names {
                h.push(',');
                h.push_str(a);
            }
            h.push('\n');
            w.write_all(h.as_bytes())?;
        }
        let mut buf = String::with_capacity(batch.ids.len() * 32);
        for i in 0..batch.ids.len() {
            use std::fmt::Write as _;
            write!(buf, "{},{}", batch.ids[i], batch.entity_type).unwrap();
            for col in &batch.attrs {
                match col {
                    agora_core::NodeColumn::Numeric(v) => {
                        let x = v[i];
                        if x.is_nan() {
                            buf.push(',');
                        } else if x == x.trunc() && x.abs() < 1e15 {
                            write!(buf, ",{}", x as i64).unwrap();
                        } else {
                            write!(buf, ",{x:.4}").unwrap();
                        }
                    }
                    agora_core::NodeColumn::Category { codes, names } => {
                        write!(buf, ",{}", names[codes[i] as usize]).unwrap();
                    }
                }
            }
            buf.push('\n');
        }
        w.write_all(buf.as_bytes())?;
        w.flush()?;
        Ok(())
    }
}
