//! Writer/reader round-trip: what the engine emits is exactly what loads back.

use agora_core::{EventBatch, EventSink};
use agora_io::read::{edge_shards, read_shard};

fn sample_batch() -> EventBatch {
    EventBatch {
        src: vec![0, 1, 2],
        dst: vec![10, 11, 12],
        t: vec![100, 200, 300],
        event_type: vec![0, 1, 0],
        label: vec![0, 0, 1],
        anomaly_id: vec![-1, -1, 7],
        attrs: vec![vec![5.5, f64::NAN, 7.25], vec![0.0, 1.0, f64::NAN]],
        attr_names: vec!["amount".into(), "channel".into()],
    }
}

fn names() -> (Vec<String>, Vec<String>, Vec<(String, Vec<String>)>) {
    (
        vec!["transfer".into(), "purchase".into()],
        vec!["normal".into(), "structuring".into()],
        vec![("channel".into(), vec!["wire".into(), "ach".into()])],
    )
}

#[test]
fn parquet_roundtrip() {
    let dir = std::env::temp_dir().join(format!("agora_pq_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (et, intents, dicts) = names();
    let mut sink = agora_io::ParquetSink::new(dir.clone(), 64, et, intents, dicts).unwrap();
    sink.write_batch(&sample_batch()).unwrap();
    sink.finish().unwrap();

    let shards = edge_shards(&dir).unwrap();
    assert_eq!(shards.len(), 1);
    let mut rows = 0usize;
    read_shard(&shards[0], &mut |b| {
        rows += b.src.len();
        assert_eq!(b.src, vec![0, 1, 2]);
        assert_eq!(b.t, vec![100, 200, 300]);
        assert_eq!(b.event_type, vec!["transfer", "purchase", "transfer"]);
        assert_eq!(b.label, vec!["normal", "normal", "structuring"]);
        let (name, amounts) = &b.numeric_attrs[0];
        assert_eq!(name, "amount");
        assert_eq!(amounts[0], 5.5);
        assert!(amounts[1].is_nan(), "null must read back as NaN");
        assert_eq!(amounts[2], 7.25);
        Ok(())
    })
    .unwrap();
    assert_eq!(rows, 3);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn csv_roundtrip() {
    let dir = std::env::temp_dir().join(format!("agora_csv_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (et, intents, dicts) = names();
    let mut sink = agora_io::CsvSink::new(dir.clone(), 64, et, intents, dicts).unwrap();
    sink.write_batch(&sample_batch()).unwrap();
    sink.finish().unwrap();

    let shards = edge_shards(&dir).unwrap();
    let mut rows = 0usize;
    read_shard(&shards[0], &mut |b| {
        rows += b.src.len();
        assert_eq!(b.dst, vec![10, 11, 12]);
        assert_eq!(b.label[2], "structuring");
        Ok(())
    })
    .unwrap();
    assert_eq!(rows, 3);
    std::fs::remove_dir_all(&dir).unwrap();
}
