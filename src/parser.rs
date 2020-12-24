//! target rule
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

use crate::ll1::{Terminal, NonTerminal, ll1};

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
            Token::Num(f64) =>2,
            Token::Add =>3,
            Token::Sub =>4,
            Token::Semicolon =>5,
            Token::EOL =>6,
            Token::LB =>7,
            Token::RB =>8,
            Token::If =>9,
            Token::Else =>10,
            Token::While =>11,
            Token::EOF=>12,
        }
    }
}
