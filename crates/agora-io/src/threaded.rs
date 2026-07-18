//! Background-writer sink: runs an inner `EventSink` on its own thread so the
//! expensive encode+write of one window overlaps generation of the next
//! (blueprint §10: I/O dominates at scale — overlap it with compute).
//!
//! The generation thread pays only a memcpy of each batch into the channel;
//! the encode (Parquet/CSV) happens off the critical path. A bounded channel
//! applies backpressure so a slow disk can't blow up memory.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

use agora_core::{EventBatch, EventSink, NodeBatch};

enum Msg {
    Batch(EventBatch),
    Nodes(NodeBatch),
}

pub struct ThreadedSink {
    tx: Option<SyncSender<Msg>>,
    handle: Option<JoinHandle<anyhow::Result<()>>>,
}

impl ThreadedSink {
    /// Wrap `inner`, moving it onto a writer thread. `depth` bounds the number
    /// of in-flight batches (backpressure).
    pub fn new<S: EventSink + Send + 'static>(mut inner: S, depth: usize) -> ThreadedSink {
        let (tx, rx): (SyncSender<Msg>, Receiver<Msg>) = sync_channel(depth.max(1));
        let handle = std::thread::Builder::new()
            .name("agora-writer".into())
            .spawn(move || -> anyhow::Result<()> {
                for msg in rx {
                    match msg {
                        Msg::Batch(b) => inner.write_batch(&b)?,
                        Msg::Nodes(n) => inner.write_nodes(&n)?,
                    }
                }
                inner.finish()
            })
            .expect("spawn writer thread");
        ThreadedSink { tx: Some(tx), handle: Some(handle) }
    }

    /// Join the writer thread, propagating any write error.
    fn join(&mut self) -> anyhow::Result<()> {
        // Drop the sender so the receiver loop ends.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            match h.join() {
                Ok(r) => r,
                Err(_) => anyhow::bail!("writer thread panicked"),
            }
        } else {
            Ok(())
        }
    }
}

impl EventSink for ThreadedSink {
    fn write_batch(&mut self, batch: &EventBatch) -> anyhow::Result<()> {
        if let Some(tx) = &self.tx {
            // One memcpy off the critical path; encode happens on the thread.
            tx.send(Msg::Batch(batch.clone()))
                .map_err(|_| anyhow::anyhow!("writer thread stopped early"))?;
        }
        Ok(())
    }

    fn write_nodes(&mut self, batch: &NodeBatch) -> anyhow::Result<()> {
        if let Some(tx) = &self.tx {
            tx.send(Msg::Nodes(batch.clone()))
                .map_err(|_| anyhow::anyhow!("writer thread stopped early"))?;
        }
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        self.join()
    }
}

impl Drop for ThreadedSink {
    fn drop(&mut self) {
        let _ = self.join();
    }
}
