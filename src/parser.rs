use crate::{Ast, Error};

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
        vec![Elem::Terminal(Token::Print), Elem::Terminal(Token::StrLit(Default::default()))],
        vec![],
    ]
}

// PRINT STRLIT EOL EOF
const TBL: [[Option<usize>; 4]; 3] = [
    [Some(0), None, Some(1), Some(5)],
    [None, None, Some(2), Some(3)],
    [Some(4), None, None,None],
];

pub fn ll1(mut tokens: Vec<Token>) -> Result<Ast, Error> {
    let rules = gen_rules();
    // 後ろから読んでいく(popしやすいので)
    tokens.reverse();
    let mut stack = Vec::new();
    stack.push(Elem::Terminal(Token::Eof));
    stack.push(Elem::Rule(Rule::Stmts));
    while !stack.is_empty() {
        let head = tokens.last().ok_or_else(|| Error::SyntaxError)?;
        println!("\nhead ({:?})", head);
        println!("stack {:?}", stack.iter().rev().collect::<Vec<&Elem>>());
        if let Elem::Terminal(token) = stack.last().unwrap() {
            if head.cardinal() != token.cardinal() {
                return Err(Error::SyntaxError);
            }
            println!("match {:?}", head);
            tokens.pop();
            stack.pop();
        } else if let Elem::Rule(rule) = stack.last().unwrap() {
            if let Some(rewriter_id) = TBL[rule.cardinal()][head.cardinal()] {
                println!("rewrite by id {}", rewriter_id);
                stack.pop();
                for elem in rules[rewriter_id].iter().rev() {
                    stack.push(elem.clone());
                }
            } else {
                return Err(Error::SyntaxError);
            }
        }
    }
    unimplemented!();
}
