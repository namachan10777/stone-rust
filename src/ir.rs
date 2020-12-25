use crate::{Ast, Internal};

#[allow(dead_code)]
pub fn compile(_: Ast) -> Internal {
    Internal {
        codes: Vec::new(),
        heap: Vec::new(),
        strs: Vec::new(),
    }
}
