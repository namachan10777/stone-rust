use crate::{Internal, Object};

#[derive(Debug, PartialEq)]
pub enum Exception {}

pub struct Vm {
    #[allow(dead_code)]
    codes: Vec<u8>,
    #[allow(dead_code)]
    regs: [Object; 6],
    #[allow(dead_code)]
    pc: u16,
    #[allow(dead_code)]
    fp: u16,
    #[allow(dead_code)]
    sp: u16,
    #[allow(dead_code)]
    ret: Object,
    #[allow(dead_code)]
    stack: [Object; 4096],
    #[allow(dead_code)]
    heap: Vec<Object>,
    #[allow(dead_code)]
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
