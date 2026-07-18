//! GraphML writer — single-file XML for igraph/Gephi/networkx interop.
//! Practical at small/medium scale; the dry-run cost model already prices its
//! ~160 B/edge footprint so users are warned before choosing it at scale.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use agora_core::{EventBatch, EventSink, NodeBatch, NodeColumn};

pub struct GraphmlSink {
    path: PathBuf,
    w: BufWriter<File>,
    event_type_names: Vec<String>,
    intent_names: Vec<String>,
    attr_dicts: Vec<(String, Vec<String>)>,
    keys_written: bool,
    attr_names: Vec<String>,
    /// Edges stream first in our pipeline, but GraphML wants nodes first;
    /// buffer edge XML and splice on finish.
    edge_buf: String,
    node_buf: String,
}

impl GraphmlSink {
    pub fn new(
        dir: PathBuf,
        event_type_names: Vec<String>,
        intent_names: Vec<String>,
        attr_dicts: Vec<(String, Vec<String>)>,
    ) -> anyhow::Result<GraphmlSink> {
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("graph.graphml");
        let w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        Ok(GraphmlSink {
            path,
            w,
            event_type_names,
            intent_names,
            attr_dicts,
            keys_written: false,
            attr_names: Vec::new(),
            edge_buf: String::new(),
            node_buf: String::new(),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl EventSink for GraphmlSink {
    fn write_batch(&mut self, b: &EventBatch) -> anyhow::Result<()> {
        use std::fmt::Write as _;
        if !self.keys_written {
            self.attr_names = b.attr_names.clone();
            self.keys_written = true;
        }
        let dicts: Vec<Option<&Vec<String>>> = b
            .attr_names
            .iter()
            .map(|n| self.attr_dicts.iter().find(|(k, _)| k == n).map(|(_, v)| v))
            .collect();
        for i in 0..b.len() {
            write!(
                self.edge_buf,
                "<edge source=\"n{}\" target=\"n{}\"><data key=\"t\">{}</data><data key=\"event_type\">{}</data><data key=\"label\">{}</data>",
                b.src[i],
                b.dst[i],
                b.t[i],
                self.event_type_names[b.event_type[i] as usize],
                self.intent_names[b.label[i] as usize],
            )
            .unwrap();
            for (c, col) in b.attrs.iter().enumerate() {
                let v = col[i];
                if v.is_nan() {
                    continue;
                }
                match dicts[c] {
                    Some(dict) => write!(
                        self.edge_buf,
                        "<data key=\"{}\">{}</data>",
                        b.attr_names[c],
                        dict[(v as usize).min(dict.len() - 1)]
                    )
                    .unwrap(),
                    None => write!(self.edge_buf, "<data key=\"{}\">{v}</data>", b.attr_names[c])
                        .unwrap(),
                }
            }
            self.edge_buf.push_str("</edge>\n");
        }
        Ok(())
    }

    fn write_nodes(&mut self, batch: &NodeBatch) -> anyhow::Result<()> {
        use std::fmt::Write as _;
        for i in 0..batch.ids.len() {
            write!(
                self.node_buf,
                "<node id=\"n{}\"><data key=\"entity_type\">{}</data>",
                batch.ids[i], batch.entity_type
            )
            .unwrap();
            for (a, col) in batch.attrs.iter().enumerate() {
                match col {
                    NodeColumn::Numeric(v) => write!(
                        self.node_buf,
                        "<data key=\"{}\">{}</data>",
                        batch.attr_names[a], v[i]
                    )
                    .unwrap(),
                    NodeColumn::Category { codes, names } => write!(
                        self.node_buf,
                        "<data key=\"{}\">{}</data>",
                        batch.attr_names[a], names[codes[i] as usize]
                    )
                    .unwrap(),
                }
            }
            self.node_buf.push_str("</node>\n");
        }
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        writeln!(self.w, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(
            self.w,
            "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">"
        )?;
        writeln!(self.w, "<key id=\"entity_type\" for=\"node\" attr.name=\"entity_type\" attr.type=\"string\"/>")?;
        for k in ["t", "event_type", "label"] {
            writeln!(self.w, "<key id=\"{k}\" for=\"edge\" attr.name=\"{k}\" attr.type=\"string\"/>")?;
        }
        for a in &self.attr_names {
            writeln!(self.w, "<key id=\"{a}\" for=\"edge\" attr.name=\"{a}\" attr.type=\"string\"/>")?;
        }
        writeln!(self.w, "<graph id=\"agora\" edgedefault=\"directed\">")?;
        self.w.write_all(self.node_buf.as_bytes())?;
        self.w.write_all(self.edge_buf.as_bytes())?;
        writeln!(self.w, "</graph>\n</graphml>")?;
        self.w.flush()?;
        Ok(())
    }
}
