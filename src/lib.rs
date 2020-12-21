#[derive(Debug, PartialEq)]
pub enum Error {
    SyntaxError,
    TypeError,
    IOError,
    InternalError,
}

pub fn entry(_: &str) -> Result<String, Error> {
    Err(Error::InternalError)
}
