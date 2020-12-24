#[derive(Debug, PartialEq)]
pub enum Token {
    Print,
    StrLit(String),
    Eol,
    Eof,
}
