use std::collections::HashMap;

/// Caution! non-terminal symbol and terminal symbol have a uniq cardinal independently.
/// e.g. Term1 -> 0, Term2 -> 1, Term3 -> 2, Rule1 -> 0, Rule2 -> 1...
pub trait Terminal {
    fn cardinal(&self) -> usize;
    fn accept(&self) -> bool;
    const N: usize;

    fn id(&self) -> SymbolId {
        SymbolId::Term(self.cardinal())
    }
}

pub trait NonTerminal {
    fn cardinal(&self) -> usize;
    const N: usize;

    fn id(&self) -> SymbolId {
        SymbolId::NTerm(self.cardinal())
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub enum SymbolId {
    Term(usize),
    NTerm(usize),
}

use std::cmp::Ordering;

impl PartialOrd<Self> for SymbolId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (SymbolId::Term(_), SymbolId::NTerm(_)) => Some(Ordering::Less),
            (SymbolId::NTerm(_), SymbolId::Term(_)) => Some(Ordering::Greater),
            (SymbolId::Term(a), SymbolId::Term(b)) => a.partial_cmp(&b),
            (SymbolId::NTerm(a), SymbolId::NTerm(b)) => a.partial_cmp(&b),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    SyntaxError,
    RuleMustBeNonTerminal,
}

pub enum ReduceSymbol<T, Ast> {
    Term(T),
    Ast(Ast),
}

pub type Rule = Vec<Vec<SymbolId>>;
// key is a cardinal of non-terminal symbol
pub type Rules<F> = HashMap<SymbolId, (Rule, F)>;

type IRule = Vec<SymbolId>;
type IRules<F> = Vec<(IRule, F)>;

type Table = Vec<Vec<Option<usize>>>;

fn gen_fluxed_and_lasts_rule(
    rules: &[Vec<SymbolId>],
) -> (Vec<SymbolId>, Option<Vec<Vec<SymbolId>>>) {
    let mut until_common_idx = 0;
    'linear_check: loop {
        let mut sample = rules[0].get(until_common_idx);
        for rule in rules {
            if rule.len() == until_common_idx || sample != rule.get(until_common_idx) {
                break 'linear_check;
            }
        }
        until_common_idx += 1;
        sample = rules[0].get(until_common_idx);
    }
    let common = rules[0][0..until_common_idx].to_vec();
    let tails = rules.iter().map(|rule| rule[until_common_idx..].to_owned());
    if tails.clone().all(|rule| rule.len() == 0) {
        (common, None)
    } else {
        (common, Some(tails.collect::<Vec<Vec<SymbolId>>>()))
    }
}

// internal impl
// 先頭が共通していれば共通部分単体のルールに書き換え、共通以後を新ルールにして追加、新ルールにも適用
fn remove_common_impl<F: std::fmt::Debug + Clone>(
    rules: &mut Vec<(Vec<Vec<SymbolId>>, F)>,
    target_idx: usize,
) {
    rules[target_idx].0.sort_by(|rule1, rule2| {
        if rule1.is_empty() {
            Ordering::Less
        } else if rule2.is_empty() {
            Ordering::Greater
        } else {
            rule1[0].partial_cmp(&rule2[0]).unwrap()
        }
    });
    let mut begin = 0;
    let mut common_removed_rule = Vec::new();
    for i in 0..rules[target_idx].0.len() {
        if rules[target_idx].0[begin].get(0) != rules[target_idx].0[i].get(0) {
            if i - begin > 1 {
                let (mut replaced, new_rule) =
                    gen_fluxed_and_lasts_rule(&rules[target_idx].0[begin..i]);
                if let Some(new_rule) = new_rule {
                    replaced.push(SymbolId::NTerm(rules.len()));
                    rules.push((new_rule, rules[target_idx].1.clone()));
                    remove_common_impl(rules, rules.len() - 1);
                }
                common_removed_rule.push(replaced);
            } else {
                common_removed_rule.push(rules[target_idx].0[begin].clone());
            }
            begin = i;
        }
    }
    if rules[target_idx].0.len() - begin > 1 {
        let (mut replaced, new_rule) =
            gen_fluxed_and_lasts_rule(&rules[target_idx].0[begin..rules[target_idx].0.len()]);
        if let Some(new_rule) = new_rule {
            replaced.push(SymbolId::NTerm(rules.len()));
            rules.push((new_rule, rules[target_idx].1.clone()));
            remove_common_impl(rules, rules.len() - 1);
        }
        common_removed_rule.push(replaced);
    } else {
        common_removed_rule.push(rules[target_idx].0[begin].clone());
    }
    rules[target_idx] = (common_removed_rule, rules[target_idx].1.clone());
}

// 共通部分削除
// A B C D | A B C E | A B F
// -> A B -> (C (D | E) | F)
fn remove_common<F: std::fmt::Debug + Clone>(
    rules: Rules<F>,
) -> Result<Vec<(Vec<Vec<SymbolId>>, F)>, Error>
where
    //F: FnMut(&mut Vec<ReduceSymbol<T, Ast>>),
{
    let mut pairs = rules
        .into_iter()
        .map(|(id, rule)| {
            if let SymbolId::NTerm(id) = id {
                Ok((id, rule))
            } else {
                Err(Error::RuleMustBeNonTerminal)
            }
        })
        .collect::<Result<Vec<(usize, (Vec<Vec<SymbolId>>, F))>, Error>>()?;
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let mut tbl = pairs
        .into_iter()
        .map(|(_, rule)| rule)
        .collect::<Vec<(Vec<Vec<SymbolId>>, F)>>();
    for i in 0..tbl.len() {
        remove_common_impl(&mut tbl, i);
    }
    Ok(tbl)
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, PartialEq)]
    enum Term {
        Num(usize),
        LP,
        RP,
        Add,
        Mul,
    }

    impl Terminal for Term {
        fn cardinal(&self) -> usize {
            match self {
                Term::Num(_) => 0,
                Term::LP => 1,
                Term::RP => 2,
                Term::Add => 3,
                Term::Mul => 4,
            }
        }

        const N: usize = 5;
        fn accept(&self) -> bool {
            if let Term::Num(_) = self {
                true
            } else {
                false
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum NTerm {
        Factor,
        Expr,
        Term,
    }

    impl NonTerminal for NTerm {
        const N: usize = 3;
        fn cardinal(&self) -> usize {
            match self {
                NTerm::Expr => 0,
                NTerm::Term => 1,
                NTerm::Factor => 2,
            }
        }
    }

    #[test]
    fn test_gen_fluxed_and_lasts_rule() {
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(1),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(1),
                SymbolId::NTerm(2),
            ],
        ]);
        assert_eq!(common, vec![SymbolId::NTerm(0), SymbolId::NTerm(0)]);
        assert_eq!(
            tails,
            Some(vec![
                vec![SymbolId::NTerm(0), SymbolId::NTerm(0)],
                vec![SymbolId::NTerm(1), SymbolId::NTerm(0)],
                vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
            ])
        );
        let (common, tails) = gen_fluxed_and_lasts_rule(&[
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
            ],
        ]);
        assert_eq!(
            common,
            vec![
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0),
                SymbolId::NTerm(0)
            ]
        );
        assert_eq!(tails, None);
    }

    #[test]
    fn test_remove_common() {
        let mut rules = HashMap::new();
        rules.insert(
            NTerm::Expr.id(),
            (
                vec![
                    vec![NTerm::Term.id()],
                    vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()],
                ],
                "expr2",
            ),
        );
        rules.insert(
            NTerm::Term.id(),
            (
                vec![
                    vec![NTerm::Factor.id()],
                    vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()],
                ],
                "term",
            ),
        );
        rules.insert(
            NTerm::Factor.id(),
            (
                vec![
                    vec![Term::Num(0).id()],
                    vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                ],
                "factor",
            ),
        );
        let removed = vec![
            (vec![vec![NTerm::Term.id(), SymbolId::NTerm(3)]], "expr2"),
            (vec![vec![NTerm::Factor.id(), SymbolId::NTerm(4)]], "term"),
            (
                vec![
                    vec![Term::Num(0).id()],
                    vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                ],
                "factor",
            ),
            (
                vec![vec![], vec![Term::Add.id(), NTerm::Expr.id()]],
                "expr2",
            ),
            (vec![vec![], vec![Term::Mul.id(), NTerm::Term.id()]], "term"),
        ];
        assert_eq!(Ok(removed), remove_common(rules));
        let mut rules = HashMap::new();
        rules.insert(
            SymbolId::NTerm(0),
            (
                vec![
                    vec![SymbolId::Term(1), SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                ],
                "hoge",
            ),
        );
        let removed = vec![
            (
                vec![
                    vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(1), SymbolId::NTerm(1)],
                ],
                "hoge",
            ),
            (
                vec![
                    vec![SymbolId::Term(2), SymbolId::Term(3)],
                    vec![SymbolId::Term(4), SymbolId::NTerm(2)],
                ],
                "hoge",
            ),
            (
                vec![vec![SymbolId::Term(3)], vec![SymbolId::Term(5)]],
                "hoge",
            ),
        ];
        assert_eq!(Ok(removed), remove_common(rules));
    }
}

pub fn ll1<T: Terminal, NT: NonTerminal, Ast, F>(
    top: NT,
    eof: T,
    rules: Rule,
    input: Vec<T>,
) -> Result<Ast, Error>
where
    F: FnMut(&mut Vec<ReduceSymbol<T, Ast>>),
{
    Err(Error::SyntaxError)
}
