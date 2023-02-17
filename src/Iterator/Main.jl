module Iterator

import Base.iterate

using ..AST
using ..Simplifier
using ..Linearization

function iterate(ast :: ASTNode)
    linearized = prepare_linearization(ast)
    if (linearized.head == And || linearized.head == Or) && length(linearized.args)==1
        linearized = linearized.args[1]
    end
    if linearized.head == And && all(isa.(linearized.args,Atom))
        return ast_to_lp(linearized), nothing
    elseif linearized.head == Or && all(map(f->f.head == And && all(isa.(f.args, Atom)), linearized.args))
        return ast_to_lp(linearized.args[1]), (2,linearized)
    else
        dnf = to_dnf(linearized)
        if dnf.head == And && length(dnf.args)==1
            dnf = dnf.args[1]
        end
        return ast_to_lp(dnf.args[1]), (2,dnf)
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