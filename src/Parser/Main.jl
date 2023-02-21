module Parser
    using Tokenize

    using Mmap
    
    include("Definitions.jl")
    include("TokenManager.jl")
    include("Parser.jl")

    export ParsingResult, DeclaredVariable, VnnExpression, CompositeVnnExpression, VnnIdentifier, VnnNumber
    export parse_file

end