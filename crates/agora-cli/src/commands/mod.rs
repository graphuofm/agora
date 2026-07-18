pub mod doctor;
pub mod domains;
pub mod generate;
pub mod init;
pub mod rules;
pub mod stats;
pub mod validate;

/// Shared output context.
pub struct Ctx {
    pub json: bool,
    pub quiet: bool,
}

impl Ctx {
    /// Print a human line unless quiet/json mode.
    pub fn say(&self, msg: impl AsRef<str>) {
        if !self.quiet && !self.json {
            println!("{}", msg.as_ref());
        }
    }
}
