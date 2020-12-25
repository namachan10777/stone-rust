use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

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

// 先頭比較で共通部分削除を実行するための実装
// あんまり順序をよく考えてない
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
    SyntaxError(String),
    MayBeLeftCyclic,
    ThisIsNotLL1(usize),
    RuleMustBeNonTerminal,
}

// Termで終端記号を保持し、Astはクロージャがメモ代わりに使う
#[derive(Debug)]
pub enum ReduceSymbol<T: fmt::Debug, Ast: fmt::Debug> {
    Term(T),
    Ast(Ast),
}

pub type Words = Vec<SymbolId>;
// ReduceSymbolを操作してAstを構築する関数
type Reducer<T, Ast> = Rc<Box<dyn Fn(&mut Vec<ReduceSymbol<T, Ast>>)>>;
// A_i -> w_ij
pub struct Rule<T: fmt::Debug, Ast: fmt::Debug> {
    pub words: Words,
    pub reducer: Reducer<T, Ast>,
}
impl<T: fmt::Debug, Ast: fmt::Debug> Clone for Rule<T, Ast> {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            reducer: self.reducer.clone(),
        }
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> fmt::Debug for Rule<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_list().entries(self.words.iter()).finish()
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> PartialEq<Rule<T, Ast>> for Rule<T, Ast> {
    fn eq(&self, other: &Self) -> bool {
        self.words.eq(&other.words)
    }
}

enum IReducer<T: fmt::Debug, Ast: fmt::Debug> {
    // 終端に付く。Rootが保持しているReducerについて、
    // どれを使うかを決める(先頭は共通化されてまとめられているので最後まで展開しないとどれで還元するか決められない）
    // 終端まで還元されてから根本が還元されるので必ずRootに対するTagは適応される
    Tag(usize),
    // RootとTagの間に埋める。何もしない
    Nop,
    // 共通部分にまとめる必要がない場合に
    Direct(Reducer<T, Ast>),
    // 一つの終端記号の言語列と還元規則の対応をそのままCloneする
    // 非効率だが実装は楽。どうせ大してメモリ食べない
    // 共通部分はまとめるだけで増えないのでこれでいける
    Root(Vec<Reducer<T, Ast>>),
}

impl<T: fmt::Debug, Ast: fmt::Debug> Clone for IReducer<T, Ast> {
    fn clone(&self) -> Self {
        match self {
            IReducer::Tag(id) => IReducer::Tag(*id),
            IReducer::Nop => IReducer::Nop,
            IReducer::Direct(f) => IReducer::Direct(f.clone()),
            IReducer::Root(map) => IReducer::Root(map.clone()),
        }
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> fmt::Debug for IReducer<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match self {
            IReducer::Tag(id) => f.write_fmt(format_args!("Tag({})", *id)),
            IReducer::Nop => f.write_str("Nop"),
            IReducer::Direct(_) => f.write_str("Direct"),
            IReducer::Root(fns) => f.write_fmt(format_args!("Fn * {}", fns.len())),
        }
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> PartialEq for IReducer<T, Ast> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (IReducer::Tag(lid), IReducer::Tag(rid)) => lid == rid,
            (IReducer::Nop, IReducer::Nop) => true,
            (IReducer::Direct(_), IReducer::Direct(_)) => true,
            (IReducer::Root(lmap), IReducer::Root(rmap)) => lmap.len() == rmap.len(),
            _ => false,
        }
    }
}

// A_i -> w_ij
struct IRule<T: fmt::Debug, Ast: fmt::Debug> {
    words: Vec<SymbolId>,
    reducer: IReducer<T, Ast>,
}
impl<T: fmt::Debug, Ast: fmt::Debug> fmt::Debug for IRule<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("IRule")
            .field("words", &self.words)
            .field("reducer", &self.reducer)
            .finish()
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> Clone for IRule<T, Ast> {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            reducer: self.reducer.clone(),
        }
    }
}

impl<T: fmt::Debug, Ast: fmt::Debug> PartialEq for IRule<T, Ast> {
    fn eq(&self, other: &Self) -> bool {
        self.words.eq(&other.words) && self.reducer.eq(&other.reducer)
    }
}

// key is a cardinal of non-terminal symbol
pub type Rules<T, Ast> = HashMap<usize, Vec<Rule<T, Ast>>>;
type IRules<T, Ast> = Vec<Vec<IRule<T, Ast>>>;

fn cmp_symbolid_option(a: Option<&SymbolId>, b: Option<&SymbolId>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, _) => Ordering::Less,
        (_, None) => Ordering::Greater,
        (Some(SymbolId::NTerm(a)), Some(SymbolId::NTerm(b))) => a.cmp(&b),
        (Some(SymbolId::Term(a)), Some(SymbolId::Term(b))) => a.cmp(&b),
        (Some(SymbolId::NTerm(_)), Some(SymbolId::Term(_))) => Ordering::Less,
        (Some(SymbolId::Term(_)), Some(SymbolId::NTerm(_))) => Ordering::Greater,
    }
}

fn cmp_head_of_rule<T: fmt::Debug, Ast: fmt::Debug>(
    a: &Rule<T, Ast>,
    b: &Rule<T, Ast>,
) -> Ordering {
    cmp_symbolid_option(a.words.get(0), b.words.get(0))
}

// 先頭いくつまで同じかを計算
fn n_sames(rules: &[(usize, Vec<SymbolId>)]) -> usize {
    let mut cnt = 0;
    if rules.is_empty() {
        return 0;
    }
    'outer: loop {
        for i in 0..rules.len() {
            if cnt >= rules[i].1.len() {
                break 'outer;
            }
            if rules[0].1[cnt] != rules[i].1[cnt] {
                break 'outer;
            }
        }
        cnt += 1;
    }
    cnt
}

type Splited<T, Ast> = (Vec<IRule<T, Ast>>, Vec<Vec<IRule<T, Ast>>>);

// 一つの終端記号について先頭でまとめ、以降を分割する
fn split_rules<T: fmt::Debug, Ast: fmt::Debug>(
    rule: &[(usize, Vec<SymbolId>)],
    nterm_offset: usize,
) -> Result<Splited<T, Ast>, Error> {
    let mut rule = rule.to_vec();
    rule.sort_by(|a, b| cmp_symbolid_option(a.1.get(0), b.1.get(0)));
    let mut new_rule = Vec::new();
    let mut added_nterm = Vec::new();
    let mut begin = 0;
    for i in 0..rule.len() + 1 {
        if i == rule.len() || rule[begin].1.get(0) != rule[i].1.get(0) {
            // 共通部分が存在する場合
            if i - begin > 1 {
                if rule[begin].1.is_empty() {
                    new_rule.push(IRule {
                        words: Vec::new(),
                        reducer: IReducer::Tag(rule[begin].0),
                    });
                } else {
                    let n_sames = n_sames(&rule[begin..i]);
                    // 全て同じ記号列の場合
                    if rule.iter().map(|r| r.1.len()).fold(0, |a, b| a.max(b)) == n_sames {
                        new_rule.push(IRule {
                            words: rule[begin].1.clone(),
                            reducer: IReducer::Tag(rule[begin].0),
                        });
                    } else {
                        let mut new_words = rule[begin].1[0..n_sames].to_vec();
                        // 新しいルールは最後に追加されるのでnterm_offsetに飛ばす
                        new_words.push(SymbolId::NTerm(nterm_offset));
                        new_rule.push(IRule {
                            words: new_words,
                            reducer: IReducer::Nop,
                        });
                        let (generated_nterm, mut generated_nterms) = split_rules::<T, Ast>(
                            &rule[begin..i]
                                .iter()
                                .map(|(tag, rule)| (*tag, rule[n_sames..rule.len()].to_vec()))
                                .collect::<Vec<(usize, Vec<SymbolId>)>>(),
                            nterm_offset + 1 + added_nterm.len(),
                        )?;
                        // 再帰的にマージ
                        added_nterm.push(generated_nterm);
                        added_nterm.append(&mut generated_nterms);
                    }
                }
            }
            // 共通部分が存在しない場合は分割は終わりなのでTagを返す
            else {
                new_rule.push(IRule {
                    words: rule[begin].1.clone(),
                    reducer: IReducer::Tag(rule[begin].0),
                });
            }
            begin = i;
        }
    }
    Ok((new_rule, added_nterm))
}

fn remove_common<T: fmt::Debug, Ast: fmt::Debug>(
    rule: Rules<T, Ast>,
) -> Result<IRules<T, Ast>, Error> {
    let mut rule_sorted = rule
        .into_iter()
        .map(|(nid, rules)| {
            (
                nid,
                rules
                    .into_iter()
                    .enumerate()
                    .collect::<Vec<(usize, Rule<T, Ast>)>>(),
            )
        })
        .collect::<Vec<(usize, Vec<(usize, Rule<T, Ast>)>)>>();
    rule_sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut new_rules = Vec::new();
    let mut n_rules = rule_sorted.len();
    new_rules.resize(n_rules, Vec::new());
    for (nidx, mut nterm) in rule_sorted {
        // Root向けにreducerの配列を作成しておく
        let reducers: Vec<Reducer<T, Ast>> = nterm
            .iter()
            .map(|rule| rule.1.reducer.clone())
            .collect::<Vec<Reducer<T, Ast>>>();
        nterm.sort_by(|a, b| cmp_head_of_rule(&a.1, &b.1));
        let (replaced, mut added) = split_rules::<T, Ast>(
            &nterm
                .into_iter()
                .map(|(idx, rule)| (idx, rule.words))
                .collect::<Vec<(usize, Vec<SymbolId>)>>(),
            n_rules,
        )?;
        let replaced = replaced
            .into_iter()
            .map(|rule| IRule {
                words: rule.words,
                // 実装の簡便化の為に書き換えで対処する
                reducer: match rule.reducer {
                    // Tagは共通部分が無いということ、
                    // 先頭について共通部分が無いのでDirectに変換
                    IReducer::Tag(idx) => IReducer::Direct(reducers[idx].clone()),
                    // Nopは共通部分がって分割した事を示すので、先頭はRoot
                    IReducer::Nop => IReducer::Root(reducers.clone()),
                    IReducer::Direct(_) => unreachable!(),
                    IReducer::Root(_) => unreachable!(),
                },
            })
            .collect::<Vec<IRule<T, Ast>>>();
        new_rules[nidx] = replaced;
        new_rules.append(&mut added);
        n_rules = new_rules.len();
    }
    Ok(new_rules)
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! map {
        { $($key:expr => $value:expr),* } => {
            {
                let mut hash = HashMap::new();
                $(
                    hash.insert($key, $value);
                )*
                hash
            }
        };
    }

    macro_rules! set {
        { } => {
            HashSet::new()
        };
        { $($value:expr),* } => {
            {
                let mut hash = HashSet::new();
                $(
                    hash.insert($value);
                )*
                hash
            }
        };
    }

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
                Term::Add => 0,
                Term::Mul => 1,
                Term::Num(_) => 2,
                Term::LP => 3,
                Term::RP => 4,
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
    fn test_remove_common() {
        let reducer: Reducer<usize, usize> =
            Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<usize, usize>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = vec![
            vec![IRule {
                words: vec![NTerm::Term.id(), SymbolId::NTerm(3)],
                reducer: IReducer::Root(vec![reducer.clone(), reducer.clone()]),
            }],
            vec![IRule {
                words: vec![NTerm::Factor.id(), SymbolId::NTerm(4)],
                reducer: IReducer::Root(vec![reducer.clone(), reducer.clone()]),
            }],
            vec![
                IRule {
                    words: vec![Term::Num(0).id()],
                    reducer: IReducer::Direct(reducer.clone()),
                },
                IRule {
                    words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
                    reducer: IReducer::Direct(reducer.clone()),
                },
            ],
            vec![
                IRule {
                    words: vec![],
                    reducer: IReducer::Tag(0),
                },
                IRule {
                    words: vec![Term::Add.id(), NTerm::Expr.id()],
                    reducer: IReducer::Tag(1),
                },
            ],
            vec![
                IRule {
                    words: vec![],
                    reducer: IReducer::Tag(0),
                },
                IRule {
                    words: vec![Term::Mul.id(), NTerm::Term.id()],
                    reducer: IReducer::Tag(1),
                },
            ],
        ];
        assert_eq!(Ok(removed), remove_common(rules));
        let mut rules = HashMap::new();
        rules.insert(
            0,
            vec![
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(1), SymbolId::Term(4), SymbolId::Term(5)],
                    reducer: reducer.clone(),
                },
                Rule {
                    words: vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: reducer.clone(),
                },
            ],
        );
        let removed = vec![
            vec![
                IRule {
                    words: vec![SymbolId::Term(0), SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: IReducer::Direct(reducer.clone()),
                },
                IRule {
                    words: vec![SymbolId::Term(1), SymbolId::NTerm(1)],
                    reducer: IReducer::Root(vec![
                        reducer.clone(),
                        reducer.clone(),
                        reducer.clone(),
                        reducer.clone(),
                        reducer.clone(),
                    ]),
                },
            ],
            vec![
                IRule {
                    words: vec![SymbolId::Term(2), SymbolId::Term(3)],
                    reducer: IReducer::Tag(0),
                },
                IRule {
                    words: vec![SymbolId::Term(4), SymbolId::NTerm(2)],
                    reducer: IReducer::Nop,
                },
            ],
            vec![
                IRule {
                    words: vec![SymbolId::Term(3)],
                    reducer: IReducer::Tag(1),
                },
                IRule {
                    words: vec![SymbolId::Term(5)],
                    reducer: IReducer::Tag(2),
                },
            ],
        ];
        assert_eq!(Ok(removed), remove_common(rules));
    }

    #[test]
    fn test_all_firsts() {
        let removed: IRules<(), ()> = vec![
            vec![
                IRule {
                    words: vec![SymbolId::NTerm(1), SymbolId::NTerm(2)],
                    reducer: IReducer::Nop,
                },
                IRule {
                    words: vec![SymbolId::NTerm(3), SymbolId::NTerm(1)],
                    reducer: IReducer::Nop,
                },
            ],
            vec![
                IRule {
                    words: vec![SymbolId::Term(0)],
                    reducer: IReducer::Nop,
                },
                IRule {
                    words: vec![],
                    reducer: IReducer::Nop,
                },
            ],
            vec![
                IRule {
                    words: vec![SymbolId::Term(1)],
                    reducer: IReducer::Nop,
                },
                IRule {
                    words: vec![],
                    reducer: IReducer::Nop,
                },
            ],
            vec![IRule {
                words: vec![SymbolId::Term(2)],
                reducer: IReducer::Nop,
            }],
        ];
        let expected = vec![
            vec![
                vec![
                    SymbolSet {
                        set: set! {0, 1},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {1},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![
                    SymbolSet {
                        set: set! {2},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {0},
                        has_eps: true,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
            ],
            vec![
                vec![
                    SymbolSet {
                        set: set! {0},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![SymbolSet {
                    set: set! {},
                    has_eps: true,
                }],
            ],
            vec![
                vec![
                    SymbolSet {
                        set: set! {1},
                        has_eps: false,
                    },
                    SymbolSet {
                        set: set! {},
                        has_eps: true,
                    },
                ],
                vec![SymbolSet {
                    set: set! {},
                    has_eps: true,
                }],
            ],
            vec![vec![
                SymbolSet {
                    set: set! {2},
                    has_eps: false,
                },
                SymbolSet {
                    set: set! {},
                    has_eps: true,
                },
            ]],
        ];
        assert_eq!(all_firsts(&removed).unwrap(), expected);
    }

    #[test]
    fn test_first() {
        let reducer: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let firsts = firsts(&removed).unwrap();
        let expected = vec![
            vec![SymbolSet {
                has_eps: false,
                set: set! {Term::Num(0).cardinal(), Term::LP.cardinal()},
            }],
            vec![SymbolSet {
                has_eps: false,
                set: set! {Term::Num(0).cardinal(), Term::LP.cardinal()},
            }],
            vec![
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Num(0).cardinal()},
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::LP.cardinal()},
                },
            ],
            vec![
                SymbolSet {
                    has_eps: true,
                    set: HashSet::new(),
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Add.cardinal()},
                },
            ],
            vec![
                SymbolSet {
                    has_eps: true,
                    set: HashSet::new(),
                },
                SymbolSet {
                    has_eps: false,
                    set: set! {Term::Mul.cardinal()},
                },
            ],
        ];
        assert_eq!(firsts, expected);
    }

    #[test]
    fn test_follows() {
        let reducer: Reducer<(), ()> = Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<(), ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let follows = follows(&removed).unwrap();
        let expected = vec![
            set! {Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::RP.cardinal()},
            set! {Term::Add.cardinal(), Term::Mul.cardinal(), Term::RP.cardinal()},
            set! {Term::RP.cardinal()},
            set! {Term::RP.cardinal(), Term::Add.cardinal()},
        ];
        assert_eq!(follows, expected);
    }

    #[test]
    fn test_gen_table() {
        let reducer: Reducer<Term, ()> =
            Rc::new(Box::new(|_: &mut Vec<ReduceSymbol<Term, ()>>| {}));
        let rules = map! {
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reducer.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reducer.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reducer.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reducer.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reducer.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reducer.clone()}
            ]
        };
        let removed = remove_common(rules).unwrap();
        let tbl = gen_table(&removed);
        let r_expr = IRule {
            words: vec![NTerm::Term.id(), SymbolId::NTerm(3)],
            reducer: IReducer::Root(vec![reducer.clone(), reducer.clone()]),
        };
        let r_term = IRule {
            words: vec![NTerm::Factor.id(), SymbolId::NTerm(4)],
            reducer: IReducer::Root(vec![reducer.clone(), reducer.clone()]),
        };
        let r_factor1 = IRule {
            words: vec![Term::Num(0).id()],
            reducer: IReducer::Direct(reducer.clone()),
        };
        let r_factor2 = IRule {
            words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()],
            reducer: IReducer::Direct(reducer.clone()),
        };
        let r_expr_2_2 = IRule {
            words: vec![],
            reducer: IReducer::Tag(0),
        };
        let r_expr_2_1 = IRule {
            words: vec![Term::Add.id(), NTerm::Expr.id()],
            reducer: IReducer::Tag(1),
        };
        let r_term_2_1 = IRule {
            words: vec![],
            reducer: IReducer::Tag(0),
        };
        let r_term_2_2 = IRule {
            words: vec![Term::Mul.id(), NTerm::Term.id()],
            reducer: IReducer::Tag(1),
        };
        let expected = vec![
            vec![None, None, Some(r_expr.clone()), Some(r_expr.clone()), None],
            vec![None, None, Some(r_term.clone()), Some(r_term.clone()), None],
            vec![
                None,
                None,
                Some(r_factor1.clone()),
                Some(r_factor2.clone()),
                None,
            ],
            vec![
                Some(r_expr_2_1.clone()),
                None,
                None,
                None,
                Some(r_expr_2_2.clone()),
            ],
            vec![
                Some(r_term_2_1.clone()),
                Some(r_term_2_2.clone()),
                None,
                None,
                Some(r_term_2_1.clone()),
            ],
        ];
        assert_eq!(Ok(expected), tbl);
    }
}

#[derive(Clone, PartialEq, Debug)]
struct SymbolSet {
    set: HashSet<usize>,
    has_eps: bool,
}
impl Default for SymbolSet {
    fn default() -> Self {
        Self {
            set: HashSet::new(),
            has_eps: false,
        }
    }
}

type AllFirstSet = Vec<Vec<Vec<SymbolSet>>>;
type FirstSet = Vec<Vec<SymbolSet>>;
type FollowSet = Vec<HashSet<usize>>;

// 各非終端記号についての各記号列において、あるインデックス以降の言語のFirst集合
fn all_firsts<T: fmt::Debug, Ast: fmt::Debug>(
    rules: &[Vec<IRule<T, Ast>>],
) -> Result<AllFirstSet, Error> {
    let mut fiw = Vec::new();
    let mut fia = Vec::new();
    // initialize
    for nterm in rules {
        fia.push(SymbolSet::default());
        let mut set_for_words = Vec::new();
        for rule in nterm {
            let mut set_for_slice = Vec::new();
            for _ in 0..rule.words.len() {
                set_for_slice.push(SymbolSet::default());
            }
            set_for_slice.push(SymbolSet {
                has_eps: true,
                set: HashSet::new(),
            });
            set_for_words.push(set_for_slice);
        }
        fiw.push(set_for_words);
    }
    // calcurate set
    let mut changed = true;
    let mut count = 0;
    while changed {
        count += 1;
        // 流石に左再帰してるでしょ（雑検知）
        if count > 10000 {
            return Err(Error::MayBeLeftCyclic);
        }
        changed = false;
        let fia_old = fia.clone();
        let fiw_old = fiw.clone();
        for (a_idx, nterm) in rules.iter().enumerate() {
            for (w_idx, rule) in nterm.iter().enumerate() {
                let mut has_eps = true;
                let mut fiw_acc = HashSet::new();
                // 後ろから舐めると計算コストを増やさずにスライスについてFirstを求められるんじゃ
                for (s_idx, w) in rule.words.iter().enumerate().rev() {
                    let fiw_i = &mut fiw[a_idx][w_idx][s_idx];
                    if let SymbolId::NTerm(id) = w {
                        // w A_i w'についてAがεを含む場合は
                        // First(A_i w')をFirst('w)とFirst(A_i)の和集合とする
                        if fia[*id].has_eps {
                            fiw_i.set = fiw_i
                                .set
                                .union(&fiw_acc)
                                .cloned()
                                .collect::<HashSet<usize>>();
                        }
                        // w A_i w'についてAがεを含まない場合は
                        // First(A_i w')はFirst(A_i)となる
                        else {
                            has_eps = false;
                        }
                        // First(A)をFirst(A w')にとりあえずマージ
                        fiw_i.set = fiw_i
                            .set
                            .union(&fia[*id].set)
                            .cloned()
                            .collect::<HashSet<usize>>();
                        fiw_acc = fiw_i.set.clone();
                    } else if let SymbolId::Term(id) = w {
                        // 終端季語なのでidのみを含む
                        fiw_acc = HashSet::new();
                        fiw_acc.insert(*id);
                        fiw_i.set = fiw_acc.clone();
                        has_eps = false;
                    }
                    fiw_i.has_eps = has_eps;
                }
                // 各非終端記号A_iについて先頭のFirstを求めてマージしていけばそれはFirst(A_i)
                // getしているのは空言語の場合があるため
                if let Some(ss) = &fiw[a_idx][w_idx].get(0) {
                    fia[a_idx].has_eps |= ss.has_eps;
                    fia[a_idx].set = fia[a_idx]
                        .set
                        .union(&ss.set)
                        .cloned()
                        .collect::<HashSet<usize>>();
                } else {
                    // 空言語しか無い場合は当然εを含む
                    fia[a_idx].has_eps = true;
                }
            }
        }
        // ここの実装をサボっています
        changed |= fia_old != fia;
        changed |= fiw_old != fiw;
    }
    Ok(fiw)
}

fn firsts<T: fmt::Debug, Ast: fmt::Debug>(rules: &[Vec<IRule<T, Ast>>]) -> Result<FirstSet, Error> {
    all_firsts(&rules).map(|firsts| {
        firsts
            .iter()
            .map(|per_rule| {
                per_rule
                    .iter()
                    .map(|per_word| per_word[0].clone())
                    .collect::<Vec<SymbolSet>>()
            })
            .collect::<Vec<Vec<SymbolSet>>>()
    })
}

/// (id of non-terminal symbol, id of rule in the non-terminal symbol) -> id of terminal symbol
fn follows<T: fmt::Debug, Ast: fmt::Debug>(
    rules: &[Vec<IRule<T, Ast>>],
) -> Result<FollowSet, Error> {
    let mut fo = FollowSet::new();
    let mut changed = true;
    fo.resize(rules.len(), Default::default());
    let firsts = all_firsts(rules)?;
    let mut count = 0;
    while changed {
        count += 1;
        if count > 10000 {
            return Err(Error::MayBeLeftCyclic);
        }
        let fo_old = fo.clone();
        for (a_idx, nterm) in rules.iter().enumerate() {
            for (w_idx, rule) in nterm.iter().enumerate() {
                for (s_idx, id) in rule.words.iter().enumerate() {
                    if let SymbolId::NTerm(id) = id {
                        // A_i -> w A_j w'
                        // という言語があった場合、Fo(A_j)にFirst(w')を追加（それはそう）
                        fo[*id] = fo[*id]
                            .union(&firsts[a_idx][w_idx][s_idx + 1].set)
                            .cloned()
                            .collect::<HashSet<usize>>();
                        // w'が空の場合Fo(A_i)を追加
                        if firsts[a_idx][w_idx][s_idx + 1].has_eps {
                            fo[*id] = fo[*id]
                                .union(&fo[a_idx])
                                .cloned()
                                .collect::<HashSet<usize>>();
                        }
                    }
                }
            }
        }
        changed = fo_old != fo;
    }
    Ok(fo)
}

type Table<T, Ast> = Vec<Vec<Option<IRule<T, Ast>>>>;

fn gen_table<T: Terminal + fmt::Debug, Ast: fmt::Debug>(
    rules: &[Vec<IRule<T, Ast>>],
) -> Result<Table<T, Ast>, Error> {
    let mut tbl = Vec::new();
    tbl.resize(rules.len(), Vec::new());
    tbl.iter_mut().for_each(|v| v.resize_with(T::N, || None));
    let firsts = firsts(&rules)?;
    let follows = follows(&rules)?;
    for nt_idx in 0..rules.len() {
        for t_idx in 0..T::N {
            for (w_idx, first_of_word) in firsts[nt_idx].iter().enumerate() {
                // T[A,a]にA -> wが入るのは
                // a ∈ Fi(w) ∨ (ε ∈ Fi(w) ∧ a ∈ Fo(A))
                if first_of_word.set.contains(&t_idx)
                    || (first_of_word.has_eps && follows[nt_idx].contains(&t_idx))
                {
                    if tbl[nt_idx][t_idx].is_some() {
                        // 複数ある場合はLL(1)ではない（なぜなら共通部分は削除済みのため）
                        return Err(Error::ThisIsNotLL1(nt_idx));
                    } else {
                        tbl[nt_idx][t_idx] = Some(rules[nt_idx][w_idx].clone());
                    }
                }
            }
        }
    }
    Ok(tbl)
}

struct Node<T: fmt::Debug, Ast: Clone + fmt::Debug> {
    reducers: Vec<Reducer<T, Ast>>,
    propagation: Option<usize>,
    child_count: usize,
}
impl<T: fmt::Debug, Ast: fmt::Debug + Clone> fmt::Debug for Node<T, Ast> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.write_fmt(format_args!(
            "({:?}, {})",
            self.propagation, self.child_count
        ))
    }
}

pub fn ll1<T: Terminal + fmt::Debug, NT: NonTerminal, Ast: Clone + fmt::Debug>(
    top: NT,
    rules: Rules<T, Ast>,
    mut input: Vec<T>,
) -> Result<Ast, Error> {
    let removed = remove_common(rules)?;
    let tbl = gen_table(&removed)?;
    input.reverse();
    let mut input_id = input
        .iter()
        .map(|tok| tok.cardinal())
        .collect::<Vec<usize>>();
    let mut ast_stack = Vec::new();
    let mut reducer_stack: Vec<Node<T, Ast>> = Vec::new();
    let mut state = Vec::new();
    state.push(top.id());
    // reduce -> reducer + 1, child_count + 1
    while !state.is_empty() {
        let top_state = state.pop().unwrap();
        // スタックの先頭が終端記号の場合
        if let SymbolId::Term(id) = top_state {
            if input_id.pop() != Some(id) {
                return Err(Error::SyntaxError(format!("input doesn't match {}", id)));
            }
            let input_top = input.pop().unwrap();
            if input_top.accept() {
                ast_stack.push(ReduceSymbol::Term(input_top));
            }
            // 還元の連鎖を開始
            // 子が全て還元されると親の還元を行う
            // 終端記号は全て還元されるのでトリガーになる
            while !reducer_stack.is_empty()
                && reducer_stack[reducer_stack.len() - 1].child_count <= 1
            {
                let i = reducer_stack.len() - 1;
                // Rootの場合は伝搬されてきたTagを使って還元を実行
                if !reducer_stack[i].reducers.is_empty() {
                    assert!(reducer_stack[i].propagation.is_some());
                    reducer_stack[i].reducers[reducer_stack[i].propagation.unwrap()](
                        &mut ast_stack,
                    );
                }
                // トップがRootへのTagの場合は伝搬させていく
                else if i > 0 {
                    assert!(reducer_stack[i - 1].propagation.is_none());
                    reducer_stack[i - 1].propagation = reducer_stack[i].propagation;
                }
                reducer_stack.pop();
            }
            // child_countを処理
            if !reducer_stack.is_empty() {
                let i = reducer_stack.len() - 1;
                reducer_stack[i].child_count -= 1;
            }
        } else if let SymbolId::NTerm(id) = top_state {
            if let Some(rewrite) = &tbl[id][*input_id
                .last()
                .ok_or_else(|| Error::SyntaxError(format!("stack remains {} {:?}", id, state)))?]
            {
                if let IReducer::Tag(tag) = rewrite.reducer {
                    // 伝搬できるようトップにセット
                    reducer_stack.push(Node {
                        reducers: vec![],
                        propagation: Some(tag),
                        child_count: rewrite.words.len(),
                    });
                } else if let IReducer::Direct(reducer) = &rewrite.reducer {
                    // 最初から伝播済み
                    reducer_stack.push(Node {
                        reducers: vec![reducer.clone()],
                        propagation: Some(0),
                        child_count: rewrite.words.len(),
                    });
                } else if let IReducer::Root(reducers) = &rewrite.reducer {
                    // 伝播待ちでReducerだけ持っている
                    reducer_stack.push(Node {
                        reducers: reducers.clone(),
                        propagation: None,
                        child_count: rewrite.words.len(),
                    });
                } else if let IReducer::Nop = &rewrite.reducer {
                    // 何もしない（伝播は還元の連鎖の中で行う）
                    reducer_stack.push(Node {
                        reducers: vec![],
                        propagation: None,
                        child_count: rewrite.words.len(),
                    });
                }
                // 空言語の場合は還元できる
                if rewrite.words.is_empty() {
                    while !reducer_stack.is_empty()
                        && reducer_stack[reducer_stack.len() - 1].child_count <= 1
                    {
                        let i = reducer_stack.len() - 1;
                        if !reducer_stack[i].reducers.is_empty() {
                            assert!(reducer_stack[i].propagation.is_some());
                            reducer_stack[i].reducers[reducer_stack[i].propagation.unwrap()](
                                &mut ast_stack,
                            );
                        } else if i > 0 {
                            assert!(reducer_stack[i - 1].propagation.is_none());
                            reducer_stack[i - 1].propagation = reducer_stack[i].propagation;
                        }
                        reducer_stack.pop();
                    }
                    if !reducer_stack.is_empty() {
                        let i = reducer_stack.len() - 1;
                        reducer_stack[i].child_count -= 1;
                    }
                }
                // 状態書き換え
                for id in rewrite.words.iter().rev() {
                    state.push(id.clone());
                }
            } else {
                return Err(Error::SyntaxError(format!(
                    "rule not found for {} {:?}",
                    id,
                    input_id.last()
                )));
            }
        }
    }
    // 最後に諸々の状態をチェックして正常終了ならOk
    if reducer_stack.is_empty() && input_id.is_empty() && ast_stack.len() == 1 {
        if let ReduceSymbol::Ast(ast) = &ast_stack[0] {
            return Ok(ast.clone());
        }
    }
    Err(Error::SyntaxError(format!(
        "Ast yet reduced completely {:?} {:?}",
        input_id,
        ast_stack.len(),
    )))
}

#[cfg(test)]
mod full_test {
    use super::*;
    macro_rules! map {
        { $($key:expr => $value:expr),* } => {
            {
                let mut hash = HashMap::new();
                $(
                    hash.insert($key, $value);
                )*
                hash
            }
        };
    }

    #[derive(Debug, PartialEq)]
    enum Term {
        Num(i64),
        LP,
        RP,
        Add,
        Mul,
        Eof,
    }

    impl Terminal for Term {
        fn cardinal(&self) -> usize {
            match self {
                Term::Add => 0,
                Term::Mul => 1,
                Term::Num(_) => 2,
                Term::LP => 3,
                Term::RP => 4,
                Term::Eof => 5,
            }
        }

        const N: usize = 6;
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
        Root,
        Expr,
        Term,
        Factor,
    }

    impl NonTerminal for NTerm {
        const N: usize = 4;
        fn cardinal(&self) -> usize {
            match self {
                NTerm::Root => 0,
                NTerm::Expr => 1,
                NTerm::Term => 2,
                NTerm::Factor => 3,
            }
        }
    }

    fn gen_rule() -> Rules<Term, i64> {
        let reduce_top: Reducer<Term, i64> = Rc::new(Box::new(|_| {}));

        let reduce_top1: Reducer<Term, i64> = Rc::new(Box::new(|_| {}));

        let reduce_top2: Reducer<Term, i64> = Rc::new(Box::new(|_| {}));
        let reduce_num: Reducer<Term, i64> = Rc::new(Box::new(|stack| {
            if let ReduceSymbol::Term(Term::Num(n)) = stack.pop().unwrap() {
                stack.push(ReduceSymbol::Ast(n));
            }
        }));
        let reduce_add: Reducer<Term, i64> = Rc::new(Box::new(|stack| {
            if let ReduceSymbol::Ast(n) = stack.pop().unwrap() {
                if let ReduceSymbol::Ast(m) = stack.pop().unwrap() {
                    stack.push(ReduceSymbol::Ast(n + m));
                }
            }
        }));
        let reduce_mul: Reducer<Term, i64> = Rc::new(Box::new(|stack| {
            if let ReduceSymbol::Ast(n) = stack.pop().unwrap() {
                if let ReduceSymbol::Ast(m) = stack.pop().unwrap() {
                    stack.push(ReduceSymbol::Ast(n * m));
                }
            }
        }));
        map! {
            NTerm::Root.cardinal()=> vec![
                Rule { words: vec![NTerm::Expr.id(), Term::Eof.id()], reducer: reduce_top.clone() },
            ],
            NTerm::Expr.cardinal()=> vec![
                Rule { words: vec![NTerm::Term.id()], reducer: reduce_top1.clone()},
                Rule { words: vec![NTerm::Term.id(), Term::Add.id(), NTerm::Expr.id()], reducer: reduce_add.clone()}
            ],
            NTerm::Term.cardinal()=>vec![
                 Rule { words: vec![NTerm::Factor.id()], reducer: reduce_top2.clone()},
                 Rule { words: vec![NTerm::Factor.id(), Term::Mul.id(), NTerm::Term.id()], reducer: reduce_mul.clone()}
            ],
            NTerm::Factor.cardinal()=>vec![
                  Rule { words: vec![Term::Num(0).id()], reducer: reduce_num.clone()},
                  Rule { words: vec![Term::LP.id(), NTerm::Expr.id(), Term::RP.id()], reducer: reduce_top2.clone()}
            ]
        }
    }

    #[test]
    fn num() {
        let input = vec![Term::Num(123), Term::Eof];
        let parsed = ll1(NTerm::Root, gen_rule(), input).unwrap();
        assert_eq!(parsed, 123);
    }

    #[test]
    fn num_paren() {
        let input = vec![Term::LP, Term::Num(123), Term::RP, Term::Eof];
        let parsed = ll1(NTerm::Root, gen_rule(), input).unwrap();
        assert_eq!(parsed, 123);
    }

    #[test]
    fn add() {
        let input = vec![Term::Num(123), Term::Add, Term::Num(111), Term::Eof];
        let parsed = ll1(NTerm::Root, gen_rule(), input).unwrap();
        assert_eq!(parsed, 234);
    }

    #[test]
    fn mul() {
        let input = vec![Term::Num(123), Term::Mul, Term::Num(111), Term::Eof];
        let parsed = ll1(NTerm::Root, gen_rule(), input).unwrap();
        assert_eq!(parsed, 13653);
    }

    #[test]
    fn complex() {
        let input = vec![
            Term::Num(7),
            Term::Mul,
            Term::LP,
            Term::Num(111),
            Term::Add,
            Term::Num(9),
            Term::RP,
            Term::Eof,
        ];
        let parsed = ll1(NTerm::Root, gen_rule(), input).unwrap();
        assert_eq!(parsed, 840);
    }
}
