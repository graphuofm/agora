//! Sharded Parquet writer — the primary output format (blueprint §12/§13).
//!
//! Layout (out_dir/):
//!   edges_00000.parquet, edges_00001.parquet, …
//!   nodes_<entity_type>.parquet
//!
//! Schema choices for zero-preprocessing loading into PyG/DGL/Neo4j/polars:
//!   src/dst   UInt64
//!   t         Int64 (unix seconds; epoch documented in agora_meta.json)
//!   event_type/label   dictionary-encoded Utf8
//!   numeric attrs      Float64, null where not applicable
//!   categorical attrs  dictionary-encoded Utf8, null where not applicable

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, StringDictionaryBuilder};
use arrow_array::types::UInt16Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use agora_core::{EventBatch, EventSink, NodeBatch, NodeColumn};

pub struct ParquetSink {
    dir: PathBuf,
    rows_per_shard: usize,
    event_type_names: Vec<String>,
    intent_names: Vec<String>,
    /// attr name -> categorical dictionary (None = numeric).
    attr_dicts: Vec<(String, Vec<String>)>,
    schema: Option<Arc<Schema>>,
    /// Per union column: Some(dict) if categorical.
    col_dicts: Option<Vec<Option<Vec<String>>>>,
    writer: Option<ArrowWriter<File>>,
    rows_in_shard: usize,
    shard_idx: u32,
    pub shards_written: Vec<PathBuf>,
}

impl ParquetSink {
    pub fn new(
        dir: PathBuf,
        shard_size_mb: u64,
        event_type_names: Vec<String>,
        intent_names: Vec<String>,
        attr_dicts: Vec<(String, Vec<String>)>,
    ) -> anyhow::Result<ParquetSink> {
        std::fs::create_dir_all(&dir)?;
        // ~24 B/edge on disk after compression (cost-model constant).
        let rows_per_shard = ((shard_size_mb * 1024 * 1024) / 24).max(1024) as usize;
        Ok(ParquetSink {
            dir,
            rows_per_shard,
            event_type_names,
            intent_names,
            attr_dicts,
            schema: None,
            col_dicts: None,
            writer: None,
            rows_in_shard: 0,
            shard_idx: 0,
            shards_written: Vec::new(),
        })
    }

    fn props() -> WriterProperties {
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(1).expect("level 1 valid")))
            .build()
    }

    fn ensure_schema(&mut self, batch: &EventBatch) {
        if self.schema.is_some() {
            return;
        }
        let dict_ty = DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
        let mut fields = vec![
            Field::new("src", DataType::UInt64, false),
            Field::new("dst", DataType::UInt64, false),
            Field::new("t", DataType::Int64, false),
            Field::new("event_type", dict_ty.clone(), false),
            Field::new("label", dict_ty.clone(), false),
            Field::new("anomaly_id", DataType::Int64, false),
        ];
        let col_dicts: Vec<Option<Vec<String>>> = batch
            .attr_names
            .iter()
            .map(|name| {
                self.attr_dicts
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, vals)| vals.clone())
            })
            .collect();
        for (i, name) in batch.attr_names.iter().enumerate() {
            let ty = if col_dicts[i].is_some() { dict_ty.clone() } else { DataType::Float64 };
            fields.push(Field::new(name, ty, true));
        }
        self.schema = Some(Arc::new(Schema::new(fields)));
        self.col_dicts = Some(col_dicts);
    }

    fn roll_writer(&mut self) -> anyhow::Result<()> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        let path = self.dir.join(format!("edges_{:05}.parquet", self.shard_idx));
        self.shard_idx += 1;
        self.rows_in_shard = 0;
        let file = File::create(&path)?;
        self.writer = Some(ArrowWriter::try_new(
            file,
            self.schema.clone().expect("schema set before roll"),
            Some(Self::props()),
        )?);
        self.shards_written.push(path);
        Ok(())
    }

    fn to_record_batch(&self, b: &EventBatch) -> anyhow::Result<RecordBatch> {
        let col_dicts = self.col_dicts.as_ref().expect("set with schema");
        let n = b.len();
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from(b.src.clone())),
            Arc::new(UInt64Array::from(b.dst.clone())),
            Arc::new(Int64Array::from(b.t.clone())),
            dict_col(&b.event_type, &self.event_type_names, n),
            dict_col(&b.label, &self.intent_names, n),
            Arc::new(Int64Array::from(b.anomaly_id.clone())),
        ];
        for (c, col) in b.attrs.iter().enumerate() {
            match &col_dicts[c] {
                Some(dict) => {
                    let mut builder = StringDictionaryBuilder::<UInt16Type>::new();
                    for &v in col {
                        if v.is_nan() {
                            builder.append_null();
                        } else {
                            let code = (v as usize).min(dict.len().saturating_sub(1));
                            builder.append_value(&dict[code]);
                        }
                    }
                    cols.push(Arc::new(builder.finish()));
                }
                None => {
                    let mut builder = Float64Builder::with_capacity(n);
                    for &v in col {
                        if v.is_nan() {
                            builder.append_null();
                        } else {
                            builder.append_value(v);
                        }
                    }
                    cols.push(Arc::new(builder.finish()));
                }
            }
        }
        Ok(RecordBatch::try_new(self.schema.clone().expect("set"), cols)?)
    }
}

fn dict_col(codes: &[u16], names: &[String], n: usize) -> ArrayRef {
    let mut builder = StringDictionaryBuilder::<UInt16Type>::with_capacity(n, names.len(), 1024);
    for &c in codes {
        builder.append_value(&names[c as usize]);
    }
    Arc::new(builder.finish())
}

impl EventSink for ParquetSink {
    fn write_batch(&mut self, batch: &EventBatch) -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        self.ensure_schema(batch);
        if self.writer.is_none() || self.rows_in_shard >= self.rows_per_shard {
            self.roll_writer()?;
        }
        let rb = self.to_record_batch(batch)?;
        self.writer.as_mut().expect("rolled").write(&rb)?;
        self.rows_in_shard += batch.len();
        Ok(())
    }

    fn write_nodes(&mut self, batch: &NodeBatch) -> anyhow::Result<()> {
        let path = self.dir.join(format!("nodes_{}.parquet", batch.entity_type));
        let mut fields = vec![Field::new("id", DataType::UInt64, false)];
        let dict_ty = DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8));
        for (i, name) in batch.attr_names.iter().enumerate() {
            let ty = match &batch.attrs[i] {
                NodeColumn::Numeric(_) => DataType::Float64,
                NodeColumn::Category { .. } => dict_ty.clone(),
            };
            fields.push(Field::new(name, ty, false));
        }
        let schema = Arc::new(Schema::new(fields));
        let mut cols: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(batch.ids.clone()))];
        for col in &batch.attrs {
            match col {
                NodeColumn::Numeric(v) => {
                    cols.push(Arc::new(arrow_array::Float64Array::from(v.clone())))
                }
                NodeColumn::Category { codes, names } => cols.push(dict_col(codes, names, codes.len())),
            }
        }
        let rb = RecordBatch::try_new(schema.clone(), cols)?;
        let mut w = ArrowWriter::try_new(File::create(&path)?, schema, Some(Self::props()))?;
        w.write(&rb)?;
        w.close()?;
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        Ok(())
    }
}
