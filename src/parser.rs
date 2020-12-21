use crate::{Ast, Error, Stmt};

struct ReduceNode<T> {
    reduced: Option<T>,
    parent: usize,
    children_size: usize,
    reducer: Option<fn(Vec<&T>) -> T>,
    children: Vec<usize>,
    remain: usize,
}

struct ReduceTree<T> {
    nodes: Vec<ReduceNode<T>>,
    current: usize,
}

enum ReduceTreeError {
    ChildrenOverflow,
}

impl<T> ReduceTree<T> {
    fn new() -> Self {
        Self {
            nodes: vec![],
            current: 0,
        }
    }

    fn finish(&self) -> Option<&T> {
        self.nodes[0].reduced.as_ref()
    }

    fn add_waiting(
        &mut self,
        reducer: fn(Vec<&T>) -> T,
        children_size: usize,
    ) -> Result<(), ReduceTreeError> {
        println!("add_waiting {} to {}", self.nodes.len(), self.current);
        if self.nodes.is_empty() {
            self.nodes.push(ReduceNode {
                reduced: None,
                parent: 0,
                reducer: Some(reducer),
                children_size,
                children: Vec::new(),
                remain: children_size,
            });
        } else {
            if self.has_all_child(self.current) {
                return Err(ReduceTreeError::ChildrenOverflow);
            }
            self.nodes.push(ReduceNode {
                reduced: None,
                parent: self.current,
                reducer: Some(reducer),
                children_size,
                children: Vec::new(),
                remain: children_size,
            });
            let child_id = self.nodes.len() - 1;
            self.nodes[self.current].children.push(child_id);
            self.current = child_id;
            self.reduce();
        }
        Ok(())
    }

    fn has_all_child(&self, id: usize) -> bool {
        self.nodes[id].children.len() == self.nodes[id].children_size
    }

    fn ready_to_reduce(&self, id: usize) -> bool {
        self.has_all_child(id) && self.nodes[id].remain == 0
    }

    fn add_term(&mut self, block: T) -> Result<(), ReduceTreeError> {
        println!("add_term {} to {}", self.nodes.len(), self.current);
        if self.nodes.is_empty() {
            self.nodes.push(ReduceNode {
                reduced: Some(block),
                parent: 0,
                reducer: None,
                children_size: 0,
                children: Vec::new(),
                remain: 0,
            });
        } else {
            self.nodes.push(ReduceNode {
                reduced: Some(block),
                reducer: None,
                parent: self.current,
                children_size: 0,
                children: Vec::new(),
                remain: 0,
            });
            let term_id = self.nodes.len() - 1;
            self.nodes[self.current].remain -= 1;
            self.nodes[self.current].children.push(term_id);
            self.reduce();
        }
        Ok(())
    }

    fn reduce(&mut self) {
        if self.ready_to_reduce(self.current) {
            println!("reduce {}", self.current);
            let children = self.nodes[self.current]
                .children
                .iter()
                .map(|c| self.nodes[*c].reduced.as_ref().unwrap())
                .collect::<Vec<&T>>();
            let reducer = self.nodes[self.current].reducer.unwrap();
            let reduced = reducer(children);
            let parent = self.nodes[self.current].parent;
            self.nodes[self.current].reduced = Some(reduced);
            if self.current != 0 {
                self.nodes[parent].remain -= 1;
                self.current = parent;
                self.reduce();
            }
        }
    }
}

#[cfg(test)]
mod test_reduce_tree {
    use super::*;

    #[derive(Debug, PartialEq, Clone)]
    struct NTree {
        id: String,
        children: Vec<NTree>,
    }

    fn reduce(children: Vec<&NTree>) -> NTree {
        let id = children
            .iter()
            .map(|nt| nt.id.as_str())
            .collect::<Vec<&str>>()
            .join(" ");
        NTree {
            id: format!("({})", id),
            children: children
                .into_iter()
                .map(|nt| nt.clone())
                .collect::<Vec<NTree>>(),
        }
    }

    #[test]
    fn test() {
        let mut tree = ReduceTree::new();
        tree.add_waiting(reduce, 2).ok();
        tree.add_term(NTree {
            id: "1".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_term(NTree {
            id: "2".to_owned(),
            children: Vec::new(),
        })
        .ok();
        assert_eq!(
            tree.nodes[0].reduced.as_ref().unwrap().id,
            "(1 2)".to_owned()
        );
        let mut tree = ReduceTree::new();
        tree.add_waiting(reduce, 2).ok();
        tree.add_term(NTree {
            id: "1".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_waiting(reduce, 3).ok();
        tree.add_term(NTree {
            id: "2".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_term(NTree {
            id: "3".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_term(NTree {
            id: "4".to_owned(),
            children: Vec::new(),
        })
        .ok();
        assert_eq!(
            tree.nodes[0].reduced.as_ref().unwrap().id,
            "(1 (2 3 4))".to_owned()
        );
        let mut tree = ReduceTree::new();
        tree.add_waiting(reduce, 2).ok();
        tree.add_waiting(reduce, 2).ok();
        tree.add_term(NTree {
            id: "1".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_term(NTree {
            id: "2".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_waiting(reduce, 2).ok();
        tree.add_term(NTree {
            id: "3".to_owned(),
            children: Vec::new(),
        })
        .ok();
        tree.add_term(NTree {
            id: "4".to_owned(),
            children: Vec::new(),
        })
        .ok();
        assert_eq!(
            tree.nodes[0].reduced.as_ref().unwrap().id,
            "((1 2) (3 4))".to_owned()
        );
    }
}

trait Cardinal {
    fn cardinal(&self) -> usize;
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Print,
    StrLit(String),
    Eol,
    Eof,
}

impl Cardinal for Token {
    fn cardinal(&self) -> usize {
        match self {
            Token::Print => 0,
            Token::StrLit(_) => 1,
            Token::Eol => 2,
            Token::Eof => 3,
        }
    }
}

/*
 * 1. stmt ::= PRINT STRLIT
 * 2. stmts ::=  stmt stmts2
 * 3. stmts2 ::= EOL stmts
 * 4. stmts2 ::=
 * 5. prog ::= stmts
 * 6. prog ::=
 */

#[derive(Clone, Debug)]
enum Rule {
    Stmts,
    Stmts2,
    Stmt,
}

impl Rule {
    fn cardinal(&self) -> usize {
        match self {
            Rule::Stmts => 0,
            Rule::Stmts2 => 1,
            Rule::Stmt => 2,
        }
    }
}

#[derive(Clone, Debug)]
enum Elem {
    Terminal(Token),
    Rule(Rule),
}

/*
 *         PRINT STRLIT EOL EOF
 * prog    1     -      -   0
 * stmts   2     -      -   -
 * stmts2  -     -      3   4
 * stmt    5     -      -   -
 */

fn gen_rules() -> [Vec<Elem>; 6] {
    [
        vec![Elem::Rule(Rule::Stmt), Elem::Rule(Rule::Stmts2)],
        vec![Elem::Terminal(Token::Eol)],
        vec![Elem::Terminal(Token::Eol), Elem::Rule(Rule::Stmts)],
        vec![],
        vec![
            Elem::Terminal(Token::Print),
            Elem::Terminal(Token::StrLit(Default::default())),
        ],
        vec![],
    ]
}

#[derive(Debug, PartialEq)]
pub enum PolyBlock {
    Term(Token),
    Stmt(Stmt),
    Ast(Ast),
}

// PRINT STRLIT EOL EOF
const TBL: [[Option<usize>; 4]; 3] = [
    [Some(0), None, Some(1), Some(5)],
    [None, None, Some(2), Some(3)],
    [Some(4), None, None, None],
];

type Reducer = fn(Vec<&PolyBlock>) -> PolyBlock;

fn stmts(children: Vec<&PolyBlock>) -> PolyBlock {
    if let [PolyBlock::Stmt(stmt), PolyBlock::Ast(ast)] = children.as_slice() {
        let mut ast = ast.clone();
        ast.push(stmt.clone());
        return PolyBlock::Ast(ast);
    } else if children.is_empty() {
        return PolyBlock::Ast(Vec::new());
    }
    unreachable!();
}

fn stmts2(children: Vec<&PolyBlock>) -> PolyBlock {
    if let [PolyBlock::Ast(ast)] = children.as_slice() {
        return PolyBlock::Ast(ast.clone());
    } else if children.is_empty() {
        return PolyBlock::Ast(Vec::new());
    }
    unimplemented!()
}

fn stmt(children: Vec<&PolyBlock>) -> PolyBlock {
    if let [PolyBlock::Term(Token::Print), PolyBlock::Term(Token::StrLit(s))] = children.as_slice()
    {
        return PolyBlock::Stmt(Stmt::Print(s.clone()));
    } else if children.is_empty() {
        return PolyBlock::Ast(Vec::new());
    }
    unimplemented!()
}

const SELECT: [bool; 4] = [true, true, false, false];

// reducer, number of children
const REDUCERS: [(Reducer, usize); 6] = [
    (stmts, 2),
    (stmts, 0),
    (stmts2, 1),
    (stmts2, 0),
    (stmt, 2),
    (stmts, 0),
];

pub fn ll1(mut tokens: Vec<Token>) -> Result<Ast, Error> {
    let rules = gen_rules();
    // 後ろから読んでいく(popしやすいので)
    tokens.reverse();
    let mut stack = Vec::new();
    stack.push(Elem::Terminal(Token::Eof));
    stack.push(Elem::Rule(Rule::Stmts));
    let mut tree = ReduceTree::new();
    while !stack.is_empty() {
        let head = tokens.last().ok_or(Error::SyntaxError)?;
        println!("\nhead ({:?})", head);
        println!("stack {:?}", stack.iter().rev().collect::<Vec<&Elem>>());
        if let Elem::Terminal(token) = stack.last().unwrap() {
            if head.cardinal() != token.cardinal() {
                return Err(Error::SyntaxError);
            }
            println!("match {:?}", head);
            if SELECT[head.cardinal()] {
                tree.add_term(PolyBlock::Term(head.clone())).ok();
            }
            tokens.pop();
            stack.pop();
        } else if let Elem::Rule(rule) = stack.last().unwrap() {
            if let Some(rewriter_id) = TBL[rule.cardinal()][head.cardinal()] {
                println!("rewrite by id {}", rewriter_id);
                stack.pop();
                let (reducer, children_size) = REDUCERS[rewriter_id];
                tree.add_waiting(reducer, children_size).ok();
                for elem in rules[rewriter_id].iter().rev() {
                    stack.push(elem.clone());
                }
            } else {
                return Err(Error::SyntaxError);
            }
        }
    }
    if let PolyBlock::Ast(ast) = tree.finish().unwrap() {
        Ok(ast.clone())
    } else {
        unreachable!();
    }
}
