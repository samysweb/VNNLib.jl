function to_linear(f :: ASTNode)
    return Map(
        (x ->
            x isa ComparisonFormula ?
                convert_comparison_to_linear(x) : x
        ),
        f)
end

function convert_comparison_to_linear(f :: ComparisonFormula)
    
end