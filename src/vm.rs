use crate::{Internal, Object};

#[derive(Debug, PartialEq)]
pub enum Exception {}

pub struct Vm {
    codes: Vec<u8>,
    regs: [Object; 6],
    pc: u16,
    fp: u16,
    sp: u16,
    ret: Object,
    stack: [Object; 4096],
    heap: Vec<Object>,
    strs: Vec<String>,
}

impl Vm {
    pub fn compile(internal: Internal) -> Self {
        Vm {
            // TODO compile Flat -> byte codes
            codes: Vec::new(),
            regs: [Object::None; 6],
            pc: 0,
            fp: 0,
            sp: 0,
            ret: Object::None,
            stack: [Object::None; 4096],
            heap: internal.heap,
            strs: internal.strs,
        }
    }

    pub fn execute(&mut self) -> Result<(), Exception> {
        Ok(())
    }
}
