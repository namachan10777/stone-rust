//!  1. primary -> "(" expr ")"
//!  2. primary -> NUMBER
//!  3. op -> "+"
//!  4. op -> "-"
//!  5. expr2 -> 空
//!  6. expr2 -> op expr
//!  7. expr -> primary expr2
//!  8. statement_opt -> 空
//!  9. statement_opt -> statement
//! 10. delim -> ";"
//! 11. delim -> EOL
//! 12. statement_list2 -> 空
//! 13. statement_list2 -> delim statement_opt statement_list2
//! 14. statement_list -> statement_opt statement_list2
//! 15. block -> "{" statement_list "}"
//! 16. simple -> expr
//! 17. else_part -> 空
//! 18. else_part -> "else" block
//! 19. statement -> "if" expr block else_part
//! 20. statement -> "while" expr block
//! 21. statement -> simple
//! 22. program -> statement_opt EOF
use crate::ll1::{ll1, NonTerminal, ReduceSymbol, Rule, Rules, Terminal};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub enum Token {
    LP,
    RP,
    Num(f64),
    Add,
    Sub,
    Semicolon,
    EOL,
    LB,
    RB,
    If,
    Else,
    While,
    EOF,
}

impl Terminal for Token {
    const N: usize = 13;
    fn accept(&self) -> bool {
        if let Token::Num(_) = self {
            return true;
        }
        return false;
    }
    fn cardinal(&self) -> usize {
        match self {
            Token::LP => 0,
            Token::RP => 1,
            Token::Num(f64) => 2,
            Token::Add => 3,
            Token::Sub => 4,
            Token::Semicolon => 5,
            Token::EOL => 6,
            Token::LB => 7,
            Token::RB => 8,
            Token::If => 9,
            Token::Else => 10,
            Token::While => 11,
            Token::EOF => 12,
        }
    }
}

#[derive(Debug, PartialEq)]
enum NonTerm {
    Primary,
    Expr,
    StmtOpt,
    StmtList2,
    Delim,
    StmtList,
    Block,
    Simple,
    ElsePart,
    Stmt,
    Program,
}

impl NonTerminal for NonTerm {
    const N: usize = 11;
    fn cardinal(&self) -> usize {
        match self {
            NonTerm::Primary => 0,
            NonTerm::Expr => 1,
            NonTerm::StmtOpt => 2,
            NonTerm::Delim => 3,
            NonTerm::StmtList2 => 4,
            NonTerm::StmtList => 5,
            NonTerm::Block => 6,
            NonTerm::Simple => 7,
            NonTerm::ElsePart => 8,
            NonTerm::Stmt => 9,
            NonTerm::Program => 10,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Num(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
enum Stmt {
    If(Expr, Vec<Stmt>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    Simple(Expr),
}

#[derive(Debug, Clone, PartialEq)]
enum Ast {
    Expr(Expr),
    StmtList(Vec<Stmt>),
}

// rewrited
//  1. primary -> "(" expr ")" | NUMBER
//  2. expr -> primary | primary "+" expr | primary "-" expr
//  3. statement_opt -> 空 | statement
//  4. statement_list2 -> delim statement_opt statement_list2
//  5. delim -> ";" | EOL
//  6. statement_list -> statement_opt delim statement_list2
//  7. block -> "{" statement_list "}"
//  8. simple -> expr
//  9. else_part -> 空 | "else" block
// 10. statement -> "if" expr block else_part | "while" expr block | simple
// 11. program -> statement EOF | EOF

fn gen_rules() -> Rules<Token, Ast> {
    let mut rules = HashMap::new();
    //  1. primary -> "(" expr ")" | NUMBER
    rules.insert(
        NonTerm::Primary.cardinal(),
        vec![
            Rule {
                words: vec![Token::LP.id(), NonTerm::Expr.id(), Token::RP.id()],
                reducer: Rc::new(Box::new(|_| {})),
            },
            Rule {
                words: vec![Token::Num(0.0).id()],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Term(Token::Num(n))) = stack.pop() {
                        stack.push(ReduceSymbol::Ast(Ast::Expr(Expr::Num(n))));
                        return;
                    }
                    unreachable!();
                })),
            },
        ],
    );
    //  2. expr -> primary | primary "+" expr | primary "-" expr
    rules.insert(
        NonTerm::Expr.cardinal(),
        vec![
            Rule {
                words: vec![NonTerm::Primary.id()],
                reducer: Rc::new(Box::new(|_| {})),
            },
            Rule {
                words: vec![NonTerm::Primary.id(), Token::Add.id(), NonTerm::Expr.id()],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::Expr(rhr))) = stack.pop() {
                        if let Some(ReduceSymbol::Ast(Ast::Expr(lhr))) = stack.pop() {
                            stack.push(ReduceSymbol::Ast(Ast::Expr(Expr::Add(
                                Box::new(lhr),
                                Box::new(rhr),
                            ))));
                            return;
                        }
                    }
                    unreachable!();
                })),
            },
            Rule {
                words: vec![NonTerm::Primary.id(), Token::Sub.id(), NonTerm::Expr.id()],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::Expr(rhr))) = stack.pop() {
                        if let Some(ReduceSymbol::Ast(Ast::Expr(lhr))) = stack.pop() {
                            stack.push(ReduceSymbol::Ast(Ast::Expr(Expr::Sub(
                                Box::new(lhr),
                                Box::new(rhr),
                            ))));
                            return;
                        }
                    }
                    unreachable!();
                })),
            },
        ],
    );
    //  3. statement_opt -> 空 | statement
    rules.insert(
        NonTerm::StmtOpt.cardinal(),
        vec![
            Rule {
                words: vec![],
                reducer: Rc::new(Box::new(|stack| {
                    println!("stmt_opt");
                    stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![])));
                })),
            },
            Rule {
                words: vec![NonTerm::Stmt.id()],
                reducer: Rc::new(Box::new(|_| {
                    println!("stmt_opt");
                })),
            },
        ],
    );
    //  4. statement_list2 -> delim statement_opt statement_list2
    rules.insert(
        NonTerm::StmtList2.cardinal(),
        vec![
            Rule {
                words: vec![
                    NonTerm::Delim.id(),
                    NonTerm::StmtOpt.id(),
                    NonTerm::StmtList2.id(),
                ],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::StmtList(mut l2))) = stack.pop() {
                        if let Some(ReduceSymbol::Ast(Ast::StmtList(mut l1))) = stack.pop() {
                            stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![])));
                        }
                    }
                })),
            },
            Rule {
                words: vec![],
                reducer: Rc::new(Box::new(|stack| {
                    stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![])));
                })),
            },
        ],
    );
    //  4. delim -> ";" | EOL
    rules.insert(
        NonTerm::Delim.cardinal(),
        vec![
            Rule {
                words: vec![Token::EOL.id()],
                reducer: Rc::new(Box::new(|stack| {})),
            },
            Rule {
                words: vec![Token::Semicolon.id()],
                reducer: Rc::new(Box::new(|stack| {})),
            },
        ],
    );

    //  6. statement_list -> statement_opt delim statement_list2
    rules.insert(
        NonTerm::StmtList.cardinal(),
        vec![Rule {
            words: vec![
                NonTerm::StmtOpt.id(),
                NonTerm::Delim.id(),
                NonTerm::StmtList2.id(),
            ],
            reducer: Rc::new(Box::new(|stack| {
                if let Some(ReduceSymbol::Ast(Ast::StmtList(mut l2))) = stack.pop() {
                    if let Some(ReduceSymbol::Ast(Ast::StmtList(mut l1))) = stack.pop() {
                        l2.append(&mut l1);
                        stack.push(ReduceSymbol::Ast(Ast::StmtList(l2)));
                    }
                }
            })),
        }],
    );
    //  6. block -> "{" EOL statement_list "}" EOL
    rules.insert(
        NonTerm::Block.cardinal(),
        vec![Rule {
            words: vec![
                Token::LB.id(),
                Token::EOL.id(),
                NonTerm::StmtList.id(),
                Token::RB.id(),
                Token::EOL.id(),
            ],
            reducer: Rc::new(Box::new(|stack| {})),
        }],
    );
    //  7. simple -> expr
    rules.insert(
        NonTerm::Simple.cardinal(),
        vec![Rule {
            words: vec![NonTerm::Expr.id()],
            reducer: Rc::new(Box::new(|stack| {})),
        }],
    );
    // 8. else_part -> 空 | "else" block
    rules.insert(
        NonTerm::ElsePart.cardinal(),
        vec![
            Rule {
                words: vec![],
                reducer: Rc::new(Box::new(|stack| {
                    stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![])));
                })),
            },
            Rule {
                words: vec![Token::Else.id(), NonTerm::Block.id()],
                reducer: Rc::new(Box::new(|_| {})),
            },
        ],
    );
    // 9. statement -> "if" expr block else_part | "while" expr block | simple
    rules.insert(
        NonTerm::Stmt.cardinal(),
        vec![
            Rule {
                words: vec![
                    Token::If.id(),
                    NonTerm::Expr.id(),
                    NonTerm::Block.id(),
                    NonTerm::ElsePart.id(),
                ],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::StmtList(else_part))) = stack.pop() {
                        if let Some(ReduceSymbol::Ast(Ast::StmtList(then_part))) = stack.pop() {
                            if let Some(ReduceSymbol::Ast(Ast::Expr(cond))) = stack.pop() {
                                stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![Stmt::If(
                                    cond, then_part, else_part,
                                )])));
                                return;
                            }
                        }
                    }
                    unreachable!()
                })),
            },
            Rule {
                words: vec![Token::While.id(), NonTerm::Expr.id(), NonTerm::Block.id()],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::StmtList(block))) = stack.pop() {
                        if let Some(ReduceSymbol::Ast(Ast::Expr(cond))) = stack.pop() {
                            stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![Stmt::While(
                                cond, block,
                            )])));
                            return;
                        }
                    }
                    unreachable!()
                })),
            },
            Rule {
                words: vec![NonTerm::Simple.id()],
                reducer: Rc::new(Box::new(|stack| {
                    if let Some(ReduceSymbol::Ast(Ast::Expr(e))) = stack.pop() {
                        stack.push(ReduceSymbol::Ast(Ast::StmtList(vec![Stmt::Simple(e)])));
                        return;
                    }
                    unreachable!()
                })),
            },
        ],
    );
    // 10. program -> statement_opt EOF
    rules.insert(
        NonTerm::Program.cardinal(),
        vec![Rule {
            words: vec![NonTerm::StmtOpt.id(), Token::EOF.id()],
            reducer: Rc::new(Box::new(|stack| {
                println!("program");
            })),
        }],
    );
    println!("{}", rules.len());
    rules
}

fn parse(src: Vec<Token>) -> Result<Option<Stmt>, super::Error> {
    let rules = gen_rules();
    let parsed = ll1(NonTerm::Program, rules, src).map_err(|e| super::Error::SyntaxError(e))?;
    if let Ast::StmtList(l) = parsed {
        if l.len() == 1 {
            return Ok(Some(l[0].clone()));
        } else {
            return Ok(None);
        }
    }
    unreachable!();
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn expr() {
        let tokens = super::super::lexer::lex("1").unwrap();
        let parsed = parse(tokens);
        assert_eq!(parsed.unwrap(), Some(Stmt::Simple(Expr::Num(1.0))));
        let tokens = super::super::lexer::lex("(1+2)-3").unwrap();
        let parsed = parse(tokens);
        assert_eq!(
            parsed.unwrap(),
            Some(Stmt::Simple(Expr::Sub(
                Box::new(Expr::Add(
                    Box::new(Expr::Num(1.0)),
                    Box::new(Expr::Num(2.0))
                )),
                Box::new(Expr::Num(3.0))
            )))
        );
    }
    #[test]
    fn if_case() {
        let tokens = super::super::lexer::lex("1").unwrap();
        let parsed = parse(tokens);
        assert_eq!(parsed.unwrap(), Some(Stmt::Simple(Expr::Num(1.0))));
        let tokens = super::super::lexer::lex("if 0 {\n1\n}\n").unwrap();
        let parsed = parse(tokens);
        assert_eq!(
            parsed.unwrap(),
            Some(Stmt::If(
                Expr::Num(0.0),
                vec![Stmt::Simple(Expr::Num(1.0))],
                vec![]
            )),
        );
    }
}
