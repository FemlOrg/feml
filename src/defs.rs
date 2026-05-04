pub(crate) const MAX_DIMS: usize = 4;
pub(crate) const MAX_SRC: usize = 10;

pub enum Status {
    AllocFailed,
    Failed,
    Success,
    Aborted,
}
