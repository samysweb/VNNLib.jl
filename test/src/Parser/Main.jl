include("Util.jl")

using Tokenize

@testset "Parser" begin
	include("TokenManager.jl")
	include("Parser.jl")
end