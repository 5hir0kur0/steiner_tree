use crate::util::NaturalOrInfinite;
use std::fmt::Display;
use std::num::ParseIntError;
use std::{error::Error, str::FromStr};

pub type NodeIndex = usize;
pub type TerminalIndex = usize;
pub type EdgeWeight = u32;

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Edge {
    to: NodeIndex,
    weight: EdgeWeight,
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub struct Graph {
    edges: Vec<Vec<Edge>>,
    terminals: Vec<NodeIndex>,
}

impl FromStr for Graph {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_graph(s)
    }
}

impl Graph {
    pub fn num_nodes(&self) -> usize {
        self.edges.len() // since `edges` is an adjacency vector this is the number of *nodes*
    }

    /// Return an iterator over all edges. Only edges `(a,b)` with `a < b` are returned since
    /// this is an undirected graph.
    pub fn edges(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex, EdgeWeight)> + '_ {
        self.edges
            .iter()
            .enumerate()
            .flat_map(|(from, e)| e.iter().map(move |&Edge { to, weight }| (from, to, weight)))
            .filter(|&(from, to, _)| from < to)
    }

    /// Iterator over the node indices.
    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> {
        0..self.num_nodes()
    }

    pub fn terminals(&self) -> &[NodeIndex] {
        &self.terminals
    }

    pub fn terminal(&self, index: TerminalIndex) -> NodeIndex {
        self.terminals[index]
    }

    pub fn num_terminals(&self) -> usize {
        self.terminals.len()
    }

    pub fn terminal_indices(&self) -> impl Iterator<Item = TerminalIndex> {
        0..self.num_terminals()
    }

    // TODO: efficient implementation
    pub fn weight(&self, from: NodeIndex, to: NodeIndex) -> NaturalOrInfinite {
        self.edges[from]
            .iter()
            .find(|x| x.to == to)
            .map(|e| e.weight)
            .map(NaturalOrInfinite::from)
            .unwrap_or_else(NaturalOrInfinite::infinity)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseError {
    line: usize,
    column: usize,
    message: String,
}

impl ParseError {
    pub fn new(line: usize, column: usize, message: String) -> Self {
        ParseError {
            line,
            column,
            message,
        }
    }
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}:{})", self.message, self.line + 1, self.column)
    }
}

impl Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;
/// Rest of input, line, column.
type ParseState<'a> = (&'a str, usize, usize);

/// Wraps a "raw", parsed [NodeIndex], i.e. a number between `1` and `#nodes`.
/// This is to avoid confusion between the 1-indexing used by the file format and the 0-based indexing used internally.
#[derive(PartialEq, Eq, Debug)]
struct ParsedNodeIndex(usize);

impl FromStr for ParsedNodeIndex {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(ParsedNodeIndex)
    }
}

impl From<ParsedNodeIndex> for NodeIndex {
    fn from(ParsedNodeIndex(i): ParsedNodeIndex) -> Self {
        assert!(
            i > 0,
            "input file does not match the PACE format (node indices must be between 1 and #nodes)"
        );
        i - 1
    }
}

/// Error if the start of the rest of the input doesn't match the expected string.
fn expect<'a>(
    (input, line, column): ParseState<'a>,
    expected: &str,
) -> ParseResult<ParseState<'a>> {
    match input.strip_prefix(expected) {
        Some(new) => Ok((new, line, column + expected.len())),
        None => Err(ParseError::new(
            line,
            column,
            format!(
                "expected '{}' but got {:?}",
                expected,
                input.split_terminator('\n').next()
            ),
        )),
    }
}

/// Expect end of input. Error if it's not the end.
fn expect_the_end(state: ParseState) -> ParseResult<ParseState> {
    let (end, line, column) = skip_whitespace(state);
    if !end.is_empty() {
        Err(ParseError::new(line, column, "expected EOL".to_string()))
    } else {
        Ok(state)
    }
}

/// Skip matches of pattern at the start. Note that this always succeeds even if the pattern doesn't
/// match (in that case, nothing is skipped).
fn skip_pattern<P: FnMut(char) -> bool>((input, line, column): ParseState, pat: P) -> ParseState {
    let trimmed = input.trim_start_matches(pat);
    (trimmed, line, column + input.len() - trimmed.len())
}

/// Skip whitespace.
fn skip_whitespace(state: ParseState) -> ParseState {
    skip_pattern(state, |c: char| c.is_ascii_whitespace() && c != '\n')
}

/// Advance state to next whitespace.
fn to_next_whitespace(state: ParseState) -> ParseResult<(&str, ParseState)> {
    let (input, line, column) = skip_whitespace(state);
    let text = input
        .split_ascii_whitespace()
        .next()
        .ok_or_else(|| ParseError::new(line, column, "unexpected end of input".to_string()))?;
    Ok((text, (&input[text.len()..], line, column + text.len())))
}

/// Like `expect` but ignores leading whitespace.
fn symbol<'a>(state: ParseState<'a>, symbol: &str) -> ParseResult<ParseState<'a>> {
    let state = skip_whitespace(state);
    expect(state, symbol)
}

/// Return the current line, i.e. from the current position until the next EOL.
fn current_line((input, line, column): ParseState) -> ParseResult<&str> {
    input
        .split('\n')
        .next()
        .ok_or_else(|| ParseError::new(line, column, "split failed".into()))
}

/// Parse anything that implements `FromStr`.
fn parse<R: FromStr>((input, line, column): ParseState<'_>) -> ParseResult<R>
where
    R::Err: Error,
{
    current_line((input, line, column))?.parse().map_err(|err| {
        ParseError::new(
            line,
            column,
            format!("could not parse input '{}': {}", input, err),
        )
    })
}

/// Parse a header of the form: `NAME : CONTENT`.
fn parse_key_value<R: FromStr>(state: ParseState, name: &str) -> ParseResult<R>
where
    R::Err: Error,
{
    let state = symbol(state, name)?;
    let state = skip_whitespace(state);
    parse(state)
}

/// Skip to the next line.
fn next_line(state: ParseState) -> ParseResult<ParseState> {
    let state = skip_pattern(state, |c| c != '\n');
    let (input, line, _) = expect(state, "\n").or_else(|_| {
        let state = skip_whitespace(state);
        expect_the_end(state)
    })?;
    Ok((input, line + 1, 0))
}

/// Check if the current line is empty (i.e. only consists of whitespace).
fn line_is_empty(state: ParseState) -> bool {
    let (text, _, _) = skip_whitespace(state);
    text.starts_with('\n')
}

/// Parse element inside single line, separated by whitespace.
fn parse_inline<T: FromStr>(state: ParseState) -> ParseResult<(T, ParseState)>
where
    T::Err: Error,
{
    let before_state = skip_whitespace(state);
    let (str, state) = to_next_whitespace(before_state)?;
    Ok((parse::<T>((str, before_state.1, before_state.2))?, state))
}

/// Parse edge in the format `E u v w`.
fn parse_edge(
    state: ParseState,
) -> ParseResult<((ParsedNodeIndex, ParsedNodeIndex, EdgeWeight), ParseState)> {
    let state = symbol(state, "E")?;
    let (from, state) = parse_inline(state)?;
    let (to, state) = parse_inline(state)?;
    let (weight, state) = parse_inline(state)?;
    let state = skip_whitespace(state);
    expect(state, "\n")?;
    let state = next_line(state)?;
    Ok(((from, to, weight), state))
}

fn skip_empty_lines(state: &mut ParseState) {
    while line_is_empty(*state) {
        let next = next_line(*state);
        if let Ok(next) = next {
            *state = next;
        } else {
            return;
        }
    }
}

/// Parse graph, reporting parse errors.
/// Since we're dealing with an NP-hard problem and thus the graphs are not going to be "huge"
/// it's acceptable to just expect the whole graph file to be read into memory.
pub fn parse_graph(text: &str) -> ParseResult<Graph> {
    let mut state = (text, 0, 0);
    skip_empty_lines(&mut state);
    state = symbol(state, "SECTION")?;
    state = symbol(state, "Graph")?;
    state = next_line(state)?;
    let num_nodes: usize = parse_key_value(state, "Nodes")?;
    state = next_line(state)?;
    let num_edges: usize = parse_key_value(state, "Edges")?;
    state = next_line(state)?;
    let mut edges = vec![vec![]; num_nodes];
    for _ in 0..num_edges {
        let ((from, to, weight), new_state) = parse_edge(state)?;
        state = new_state;
        let from = NodeIndex::from(from);
        let to = NodeIndex::from(to);
        if !edges[from].iter().any(|e: &Edge| e.to == to) {
            edges[from].push(Edge { to, weight });
            edges[to].push(Edge { to: from, weight })
        }
    }
    state = symbol(state, "END")?;
    skip_empty_lines(&mut state);
    state = symbol(state, "SECTION")?;
    state = symbol(state, "Terminals")?;
    state = next_line(state)?;
    let num_terminals: usize = parse_key_value(state, "Terminals")?;
    state = next_line(state)?;
    let mut terminals = vec![];
    for _ in 0..num_terminals {
        let terminal: ParsedNodeIndex = parse_key_value(state, "T")?;
        if terminal.0 > num_nodes {
            return Err(ParseError::new(
                state.1,
                2,
                "invalid terminal (too big)".into(),
            ));
        }
        let terminal = NodeIndex::from(terminal);
        if !terminals.contains(&terminal) {
            terminals.push(terminal);
        }
        state = next_line(state)?;
    }
    state = symbol(state, "END")?;
    skip_empty_lines(&mut state);
    symbol(state, "EOF")?;
    terminals.sort_unstable();
    for vec in &mut edges {
        vec.sort_unstable_by_key(|e| e.to);
    }
    Ok(Graph { edges, terminals })
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::util::TestResult;

    use super::*;

    #[test]
    fn test_parsed_node_index() -> TestResult {
        let p: ParsedNodeIndex = "42".parse()?;
        assert_eq!(p, ParsedNodeIndex(42));
        assert_eq!(NodeIndex::from(p), 41);
        Ok(())
    }

    #[test]
    fn test_expect() {
        let state = ("hello", 0, 0);
        assert_eq!(expect(state, "hello"), Ok(("", 0, 5)));
        assert!(expect(state, "hallo").is_err());
    }

    #[test]
    fn test_skip_whitespace() {
        let state = ("   \thello", 0, 0);
        let state = skip_whitespace(state);
        assert_eq!(state, ("hello", 0, 4));
        let state = skip_whitespace(state);
        assert_eq!(state, ("hello", 0, 4));
    }

    #[test]
    fn test_to_next_whitespace() -> ParseResult<()> {
        let state = ("\thello world", 0, 0);
        let (res, state) = to_next_whitespace(state)?;
        assert_eq!(res, "hello");
        assert_eq!(state, (" world", 0, 6));
        Ok(())
    }

    #[test]
    fn test_symbol() -> ParseResult<()> {
        let state = ("\thello world", 0, 0);
        assert!(symbol(state, "hallo").is_err());
        let state = symbol(state, "hello")?;
        let _ = symbol(state, "world")?;
        Ok(())
    }

    #[test]
    fn test_parse() -> ParseResult<()> {
        let state = ("42", 0, 0);
        let parsed: usize = parse(state)?;
        assert_eq!(parsed, 42);
        let state = ("ff", 0, 0);
        assert!(parse::<usize>(state).is_err());
        Ok(())
    }

    /// ```text
    ///     1
    /// 0 ----- 1
    ///  \     /
    /// 3 \   / 2
    ///    \ /
    ///     2
    /// ```
    /// Terminals: `0, 2`
    pub(crate) fn small_test_graph() -> ParseResult<Graph> {
        "SECTION Graph\n\
        Nodes 3\n\
        Edges 3\n\
        E 1 2 1\n\
        E 2 3 2\n\
        E 3 1 3\n\
        END\n\

        SECTION Terminals\n\
        Terminals 2\n\
        T 1\n\
        T 3\n\
        END\n\

        EOF\n"
            .parse::<Graph>()
    }

    /// ```text
    ///    1
    ///  0----1
    ///  |  / |
    /// 7| /1 |2
    ///  |/   |
    ///  2----3
    ///    4
    /// ```
    pub(crate) fn shortcut_test_graph() -> ParseResult<Graph> {
        "SECTION Graph\n\
        Nodes 4\n\
        Edges 5\n\
        E 2 1 1\n\
        E 2 4 2\n\
        E 2 3 1\n\
        E 4 3 4\n\
        E 1 3 7\n\
        END\n\

        SECTION Terminals\n\
        Terminals 2\n\
        T 1\n\
        T 3\n\
        END\n\

        EOF\n"
            .parse::<Graph>()
    }

    /// From [Wikipedia](https://de.wikipedia.org/wiki/Steinerbaumproblem#/media/Datei:Steinerbaum_Beispiel_Graph.svg).
    pub(crate) fn steiner_example_wiki() -> ParseResult<Graph> {
        "SECTION Graph\n\
        Nodes 12\n\
        Edges 15\n\
        E 1 2 15\n\
        E 2 3 30\n\
        E 3 4 50\n\
        E 4 7 30\n\
        E 1 5 25\n\
        E 2 9 50\n\
        E 2 6 45\n\
        E 3 6 40\n\
        E 6 8 60\n\
        E 7 8 20\n\
        E 5 9 30\n\
        E 9 11 15\n\
        E 8 10 50\n\
        E 11 10 40\n\
        E 12 11 10\n\
        END\n\

        SECTION Terminals\n\
        Terminals 5\n\
        T 1\n\
        T 9\n\
        T 12\n\
        T 7\n\
        T 8\n\
        END\n\

        EOF\n"
            .parse::<Graph>()
    }

    /// Example from the original paper by Dreyfus & Wagner.
    pub(crate) fn steiner_example_paper() -> ParseResult<Graph> {
        "SECTION Graph\n\
        Nodes 7\n\
        Edges 21\n\
        E 1 2 2\n\
        E 1 3 2\n\
        E 1 4 2\n\
        E 1 5 1\n\
        E 1 6 1\n\
        E 1 7 2\n\
        E 2 3 2\n\
        E 2 4 2\n\
        E 2 5 2\n\
        E 2 6 1\n\
        E 2 7 2\n\
        E 3 4 2\n\
        E 3 5 2\n\
        E 3 6 2\n\
        E 3 7 1\n\
        E 4 5 2\n\
        E 4 6 1\n\
        E 4 7 1\n\
        E 5 6 2\n\
        E 5 7 1\n\
        E 6 7 1\n\
        END\n\

        SECTION Terminals\n\
        Terminals 4\n\
        T 1\n\
        T 2\n\
        T 3\n\
        T 4\n\
        END\n\

        EOF\n"
            .parse::<Graph>()
    }

    #[test]
    fn test_edges() -> TestResult {
        let short = shortcut_test_graph()?;
        let mut edges = short.edges().collect::<Vec<_>>();
        edges.sort_by_key(|&(a, b, _)| [a, b]);
        assert_eq!(
            edges,
            vec![(0, 1, 1), (0, 2, 7), (1, 2, 1), (1, 3, 2), (2, 3, 4),]
        );
        Ok(())
    }

    #[test]
    fn test_parse_small_graph() -> TestResult {
        let graph = small_test_graph()?;
        assert_eq!(
            graph,
            Graph {
                edges: vec![
                    vec![Edge { to: 1, weight: 1 }, Edge { to: 2, weight: 3 },],
                    vec![Edge { to: 0, weight: 1 }, Edge { to: 2, weight: 2 },],
                    vec![Edge { to: 0, weight: 3 }, Edge { to: 1, weight: 2 },],
                ],
                terminals: vec![0, 2],
            }
        );

        Ok(())
    }

    #[test]
    fn test_weight() -> TestResult {
        let graph = shortcut_test_graph()?;
        assert_eq!(graph.weight(0, 2), 7.into());
        assert_eq!(graph.weight(2, 0), 7.into());
        assert_eq!(graph.weight(2, 3), 4.into());
        assert_eq!(graph.weight(3, 2), 4.into());
        assert_eq!(graph.weight(3, 0), NaturalOrInfinite::infinity());
        assert_eq!(graph.weight(0, 3), NaturalOrInfinite::infinity());
        Ok(())
    }
}
