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
    if dnf.head == And && length(dnf.args)==1
        dnf = dnf.args[1]
    end
    if dnf.head == Or
        return ast_to_lp(dnf.args[1]), (2,dnf)
    else
        return ast_to_lp(dnf), nothing
    end
end

function iterate(ast :: ASTNode, state :: Nothing)
    return nothing
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