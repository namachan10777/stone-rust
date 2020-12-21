use std::fs;

mod ir;
mod lexer;
mod parser;
mod vm;

#[derive(Debug, PartialEq)]
pub enum Error {
    SyntaxError,
    TypeError,
    IOError,
    InternalError,
    RuntimeError(vm::Exception),
}

#[derive(Debug, PartialEq)]
pub enum Ast {
    Print(String),
}

#[derive(Debug, PartialEq)]
pub enum Flat {
    // u8で指定されたレジスタ中の文字列を出力
    Print(u8),
    // u16で指定された文字列領域の文字列リテラルをregに格納
    SConst(u16, u8),
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
    let ast = parser::root(&src)?;
    let flat = ir::compile(ast);
    let mut vm = vm::Vm::compile(flat);
    vm.execute()
        .map_err(|exception| Error::RuntimeError(exception))
}
