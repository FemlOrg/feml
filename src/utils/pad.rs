pub const FEML_MEM_ALIGN: usize = 16;

#[macro_export]
macro_rules! feml_pad {
    ($x:expr, $n: expr) => {
        ($x + $n - 1) & !($n - 1)
    };
}
