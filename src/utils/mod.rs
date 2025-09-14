pub mod log;
pub mod pad;

pub use pad::FEML_MEM_ALIGN;

#[derive(Debug, Clone, Copy)]
pub enum FemlLogLevel {
    None,
    Debug,
    Info,
    Warn,
    Error,
    Cont, // continue previous log
}

// fn fn_name_of<T>(_: T) -> &'static str {
//     std::any::type_name::<T>()
// }

// macro_rules! fn_name {
//     () => {{
//         fn f() {}
//         let full = fn_name_of(f);
//         &full[..full.len() - 3]
//     }};
// }

// fn feml_log_internal(log_level: FemlLogLevel, file: &str, line: u32, args: fmt::Arguments) {
//     eprintln!("{} : {} : {} : {}", file, line, fn_name!(), args);
// }