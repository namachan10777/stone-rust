use crate::parser::Token;
use crate::Error;

fn match_str(src: &str, compare: &str) -> Option<usize> {
    if compare.len() > src.len() {
        return None;
    }
    if &src[0..compare.len()] == compare {
        Some(compare.len())
    } else {
        None
    }
}

fn match_strlit(src: &str) -> Option<usize> {
    if src.len() < 2 || &src[0..1] != "\"" {
        return None;
    }
    let mut cnt = 1;
    loop {
        if cnt >= src.len() {
            return None;
        }
        if cnt <= src.len() - 2 {
            if let "\\\"" | "\\\\" = &src[cnt..cnt + 2] {
                cnt += 2;
                continue;
            } else if let "\"" = &src[cnt..cnt + 1] {
                return Some(cnt + 1);
            }
        } else if let "\"" = &src[cnt..cnt + 1] {
            return Some(cnt + 1);
        }
        cnt += 1;
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
        Some(cnt)
    } else {
        None
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
        Some(cnt)
    } else {
        None
    }
}

fn match_ident(src: &str) -> Option<usize> {
    if src.is_empty() {
        return None;
    }
    for (idx, c) in src.chars().enumerate() {
        if !c.is_alphabetic() {
            if idx == 0 {
                return None;
            }
            return Some(idx);
        }
    }
    Some(src.len())
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
        } else if let Some(step) = match_strlit(remain) {
            tokens.push(Token::Str(src[begin + 1..begin + step - 2].to_string()));
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
        } else if let Some(step) = match_str(remain, "*") {
            tokens.push(Token::Mul);
            begin += step;
        } else if let Some(step) = match_str(remain, "/") {
            tokens.push(Token::Div);
            begin += step;
        } else if let Some(step) = match_str(remain, "==") {
            tokens.push(Token::Equal);
            begin += step;
        } else if let Some(step) = match_str(remain, ">") {
            tokens.push(Token::Gret);
            begin += step;
        } else if let Some(step) = match_str(remain, "<") {
            tokens.push(Token::Less);
            begin += step;
        } else if let Some(step) = match_str(remain, "&&") {
            tokens.push(Token::And);
            begin += step;
        } else if let Some(step) = match_str(remain, "||") {
            tokens.push(Token::Or);
            begin += step;
        } else if let Some(step) = match_str(remain, "=") {
            tokens.push(Token::Assign);
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
        } else if let Some(step) = match_ident(remain) {
            tokens.push(Token::Var(src[begin..begin + step].to_string()));
            begin += step;
        } else if let Some(step) = match_str(remain, "\n") {
            tokens.push(Token::EOL);
            begin += step;
        } else {
            return Err(Error::LexerError(begin));
        }
    }
    while tokens.last() == Some(&Token::EOL) {
        tokens.pop();
    }
    tokens.push(Token::EOL);
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
                Token::EOL,
                Token::EOF
            ])
        );
    }

    #[test]
    fn strlit() {
        assert_eq!(match_strlit("\"hoge\""), Some(6));
        assert_eq!(match_strlit("\"hoge"), None);
        assert_eq!(match_strlit("\"ho\\\"ge\""), Some(8));
    }

    #[test]
    fn ident() {
        assert_eq!(match_ident("hoge &"), Some(4));
    }
}
