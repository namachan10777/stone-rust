use crate::parser::Token;
use crate::Error;

const PRINT: &str = "print";

fn match_str(src: &str, compare: &str) -> Option<usize> {
    if compare.len() > src.len() {
        return None;
    }
    if &src[0..compare.len()] == compare {
        return Some(compare.len());
    } else {
        return None;
    }
}

// TODO 実装サボっており整数のみしか扱えません
fn match_num(src: &str) -> Option<usize> {
    if src.is_empty() {
        return None;
    }
    let mut cnt = 0;
    while cnt < src.len() {
        // めちゃくちゃバカっぽいコードに見えますが、Rustの仕様だと多分これが一番速い
        match &src[cnt..cnt + 1] {
            "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" => {}
            _ => {
                break;
            }
        }
        cnt += 1;
    }
    if cnt != 0 {
        return Some(cnt);
    } else {
        return None;
    }
}

fn match_white(src: &str) -> Option<usize> {
    if src.is_empty() {
        return None;
    }
    let mut cnt = 0;
    while cnt < src.len() {
        // めちゃくちゃバカっぽいコードに見えますが、Rustの仕様だと多分これが一番速い
        match &src[cnt..cnt + 1] {
            " " | "\t" | "\r" => {}
            _ => {
                break;
            }
        }
        cnt += 1;
    }
    if cnt != 0 {
        return Some(cnt);
    } else {
        return None;
    }
}

pub fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let mut tokens = Vec::new();
    let mut begin = 0;
    while begin < src.len() {
        let remain = &src[begin..src.len()];
        if let Some(step) = match_white(remain) {
            begin += step;
        } else if let Some(step) = match_num(remain) {
            tokens.push(Token::Num(src[begin..begin + step].parse().unwrap()));
            begin += step;
        } else if let Some(step) = match_str(remain, "(") {
            tokens.push(Token::LP);
            begin += step;
        } else if let Some(step) = match_str(remain, ")") {
            tokens.push(Token::RP);
            begin += step;
        } else if let Some(step) = match_str(remain, "+") {
            tokens.push(Token::Add);
            begin += step;
        } else if let Some(step) = match_str(remain, "-") {
            tokens.push(Token::Sub);
            begin += step;
        } else if let Some(step) = match_str(remain, ";") {
            tokens.push(Token::Semicolon);
            begin += step;
        } else if let Some(step) = match_str(remain, "{") {
            tokens.push(Token::LB);
            begin += step;
        } else if let Some(step) = match_str(remain, "}") {
            tokens.push(Token::RB);
            begin += step;
        } else if let Some(step) = match_str(remain, "if") {
            tokens.push(Token::If);
            begin += step;
        } else if let Some(step) = match_str(remain, "else") {
            tokens.push(Token::Else);
            begin += step;
        } else if let Some(step) = match_str(remain, "while") {
            tokens.push(Token::While);
            begin += step;
        } else if let Some(step) = match_str(remain, "\n") {
            tokens.push(Token::EOL);
            begin += step;
        } else {
            return Err(Error::LexerError);
        }
    }
    tokens.push(Token::EOF);
    Ok(tokens)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test() {
        let src = "if 123 else\n  } {";
        assert_eq!(
            lex(src),
            Ok(vec![
                Token::If,
                Token::Num(123.0),
                Token::Else,
                Token::EOL,
                Token::RB,
                Token::LB,
                Token::EOF
            ])
        );
    }
}
