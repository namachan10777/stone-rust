use std::fs;

mod ir;
mod lexer;
mod ll1;
mod parser;
mod vm;

#[derive(Debug, PartialEq)]
pub enum Error {
    LexerError(usize),
    SyntaxError(ll1::Error),
    TypeError,
    IOError,
    InternalError,
    RuntimeError(vm::Exception),
}

pub type Ast = Vec<Stmt>;
#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Print(String),
}

#[derive(Debug, PartialEq)]
pub enum Flat {
    // u8で指定されたレジスタ中の文字列を出力
    Print(u8),
    // u16で指定された文字列領域の文字列リテラルをregに格納 SConst(u16, u8),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Object {
    // strsへの参照
    Str(u16),
    None,
}

#[derive(Debug, PartialEq)]
pub struct Internal {
    codes: Vec<Flat>,
    strs: Vec<String>,
    heap: Vec<Object>,
}

pub fn entry(src_path: &str) -> Result<(), Error> {
    let src = fs::read_to_string(src_path).map_err(|_| Error::IOError)?;
    let tokens = lexer::lex(&src)?;
    let ast = parser::parse(tokens)?;
    println!("{:#?}", ast);
    Ok(())
}
