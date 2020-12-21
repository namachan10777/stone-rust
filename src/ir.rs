use crate::{Ast, Internal};

pub fn compile(_: Ast) -> Internal {
    Internal {
        codes: Vec::new(),
        heap: Vec::new(),
        strs: Vec::new(),
    }
}
