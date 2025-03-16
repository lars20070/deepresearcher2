Generate UML class diagrams with `pyreverse` from the `pylint` package and `dot` from the `graphviz` package. The UML diagrams are oriented left to right with the `rankdir=LR` option.
```bash
# brew install graphviz

uv run pyreverse -o dot -A -d ./uml ./src/deepresearcher
dot -Tpng -Grankdir=LR -o ./uml/classes.png ./uml/classes.dot
dot -Tpng -Grankdir=LR -o ./uml/packages.png ./uml/packages.dot

# uv run uml
```
