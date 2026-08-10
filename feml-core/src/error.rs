//! Compact error type with context chaining.
//!
//! Kept deliberately small: a kind + optional context chain. No per-backend
//! variants in core (backends wrap their own errors via `Msg`).

use crate::dtype::DType;
use std::borrow::Cow;
use std::fmt;

#[derive(Debug)]
pub enum ErrorKind {
    /// Generic message (static or owned).
    Msg(Cow<'static, str>),
    /// Shape/layout mismatch with a descriptive message.
    Shape(String),
    /// Operation not supported for a dtype.
    UnsupportedDType { dtype: DType, op: &'static str },
    /// Backend error, already rendered to a message.
    Backend(&'static str, String),
}

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    context: Vec<Cow<'static, str>>,
}

impl Error {
    pub fn msg(msg: impl Into<Cow<'static, str>>) -> Self {
        Self { kind: ErrorKind::Msg(msg.into()), context: Vec::new() }
    }

    pub fn shape(msg: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Shape(msg.into()), context: Vec::new() }
    }

    pub fn unsupported_dtype(dtype: DType, op: &'static str) -> Self {
        Self { kind: ErrorKind::UnsupportedDType { dtype, op }, context: Vec::new() }
    }

    pub fn backend(backend: &'static str, msg: impl Into<String>) -> Self {
        Self { kind: ErrorKind::Backend(backend, msg.into()), context: Vec::new() }
    }

    /// Append context; the oldest context is printed first.
    pub fn context(mut self, ctx: impl Into<Cow<'static, str>>) -> Self {
        self.context.push(ctx.into());
        self
    }

    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        for ctx in &self.context {
            write!(f, "\n  context: {ctx}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Msg(m) => write!(f, "{m}"),
            ErrorKind::Shape(m) => write!(f, "shape error: {m}"),
            ErrorKind::UnsupportedDType { dtype, op } => {
                write!(f, "dtype {} not supported for op {op}", dtype.name())
            }
            ErrorKind::Backend(name, msg) => write!(f, "{name} backend error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn display_msg() {
        assert_eq!(Error::msg("boom").to_string(), "boom");
    }

    #[test]
    fn display_shape() {
        assert_eq!(Error::shape("dim mismatch").to_string(), "shape error: dim mismatch");
    }

    #[test]
    fn context_chain_orders_oldest_first() {
        let e = Error::msg("root").context("mid").context("leaf");
        let s = e.to_string();
        assert!(s.contains("root"));
        let mid = s.find("mid").unwrap();
        let leaf = s.find("leaf").unwrap();
        assert!(mid < leaf, "oldest context should print first");
    }

    #[test]
    fn source_none() {
        assert!(Error::msg("x").source().is_none());
    }
}
