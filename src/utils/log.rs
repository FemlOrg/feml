use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum FemlLogLevel {
    None,
    Debug,
    Info,
    Warn,
    Error,
    Cont, // continue previous log
}

#[macro_export]
macro_rules! feml_abort {
    ($($arg:tt)*) => {
        panic!("ABORT at {} : {} : {}", file!(), line!(), format_args!($($arg)*));
    };
}

#[macro_export]
macro_rules! feml_debug {
    ($($arg:tt)*) => {
        $crate::utils::log::feml_log_internal(
            $crate::utils::log::FemlLogLevel::Debug,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! feml_info {
    ($($arg:tt)*) => {
        $crate::utils::log::feml_log_internal(
            $crate::utils::log::FemlLogLevel::Info,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! feml_warn {
    ($($arg:tt)*) => {
        $crate::utils::log::feml_log_internal(
            $crate::utils::log::FemlLogLevel::Warn,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

#[macro_export]
macro_rules! feml_error {
    ($($arg:tt)*) => {
        $crate::utils::log::feml_log_internal(
            $crate::utils::log::FemlLogLevel::Error,
            file!(),
            line!(),
            format_args!($($arg)*),
        )
    };
}

fn fn_name_of<T>(_: T) -> &'static str {
    std::any::type_name::<T>()
}

macro_rules! fn_name {
    () => {{
        fn f() {}
        let full = fn_name_of(f);
        &full[..full.len() - 3]
    }};
}

#[allow(unused)]
pub fn feml_log_internal(log_level: FemlLogLevel, file: &str, line: u32, args: fmt::Arguments) {
    eprintln!("{} : {} : {} : {}", file, line, fn_name!(), args);
}

pub fn feml_print_backtrace() {
    let bt = std::backtrace::Backtrace::force_capture();
    eprintln!("{bt}");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_log() {
        feml_warn!("test_log");
    }
}
