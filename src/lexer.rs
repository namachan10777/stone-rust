use crate::parser::Token;
use crate::Error;

const PRINT: &'static str = "print";

fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let mut tokens = Vec::new();
    let mut beg = 0;
    loop {
        if beg == src.len() {
            tokens.push(Token::Eof);
            break;
        } else if (src.len() - beg) >= PRINT.len() && src[beg..beg + PRINT.len()].eq(PRINT) {
            beg += PRINT.len();
            tokens.push(Token::Print);
        } else if src[beg..beg + 1].eq(" ")
            || src[beg..beg + 1].eq("\t")
            || src[beg..beg + 1].eq("\r")
        {
            // skip
            beg += 1;
        } else if src[beg..beg + 1].eq("\n") {
            tokens.push(Token::Eol);
            beg += 1;
        } else if (src.len() - beg) >= 2 && src[beg..beg + 1].eq("\"") {
            let mut s = String::new();
            beg += 1;
            let mut end = beg;
            loop {
                if (src.len() - end) >= 2 && src[end..end + 1].eq("\\") {
                    s.push_str(&src[beg..end]);
                    match &src[end + 1..end + 2] {
                        "\"" | "\\" => s.push_str(&src[end + 1..end + 2]),
                        "n" => s.push_str("\n"),
                        _ => return Err(Error::SyntaxError),
                    }
                    end += 2;
                    beg = end;
                } else if src.len() - end >= 1 && src[end..end + 1].eq("\n") {
                    return Err(Error::SyntaxError);
                } else if src.len() - end >= 1 && src[end..end + 1].eq("\"") {
                    s.push_str(&src[beg..end]);
                    tokens.push(Token::StrLit(s));
                    end += 1;
                    beg = end;
                    break;
                } else if src.len() - end <= 0 {
                    return Err(Error::SyntaxError);
                } else {
                    end += 1;
                }
            }
        } else {
            return Err(Error::SyntaxError);
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_lex() {
        assert_eq!(lex("print"), Ok(vec![Token::Print, Token::Eof]));
        assert_eq!(
            lex("\"hoge\""),
            Ok(vec![Token::StrLit("hoge".to_owned()), Token::Eof])
        );
        assert_eq!(
            lex("\"hoge\\\"fuga\""),
            Ok(vec![Token::StrLit("hoge\"fuga".to_owned()), Token::Eof])
        );
        assert_eq!(
            lex("\"hoge\\nfuga\""),
            Ok(vec![Token::StrLit("hoge\nfuga".to_owned()), Token::Eof])
        );
        assert_eq!(
            lex("print \"hoge\\nfuga\""),
            Ok(vec![
                Token::Print,
                Token::StrLit("hoge\nfuga".to_owned()),
                Token::Eof
            ])
        );
        assert_eq!(
            lex("print \"hoge\"\nprint \"fuga\""),
            Ok(vec![
                Token::Print,
                Token::StrLit("hoge".to_owned()),
                Token::Eol,
                Token::Print,
                Token::StrLit("fuga".to_owned()),
                Token::Eof
            ])
        );
    }
}
