use crate::{Ast, Error};

#[derive(Debug, PartialEq)]
pub enum Token {
    Print,
    StrLit(String),
    Eol,
    Eof,
}

pub fn root(_: &str) -> Result<Ast, Error> {
    unimplemented!()
}
