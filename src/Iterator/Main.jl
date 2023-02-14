module Iterator

import Base.iterate

using ..AST
using ..Simplifier
using ..Linearization

function iterate(ast :: ASTNode)
    # Adjust atoms and terms
    linearized = prepare_linearization(ast)
    # Adjust formula structure
    dnf = to_dnf(linearized)
    if dnf.head == And && dnf.args[1] isa CompositeFormula && dnf.args[1].head == Or
        dnf = dnf.args[1]
    end
    return ast_to_lp(dnf.args[1]), (2,dnf)
end

function iterate(ast :: ASTNode, state :: Tuple{Int64,ASTNode})
    i,ast = state
    if i > length(ast.args)
        return nothing
    else
        return ast_to_lp(ast.args[i]), (i+1,ast)
    end
end

end # module Iterator