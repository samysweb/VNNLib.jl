function create_tm(input :: String) :: VNNLib.Parser.TokenManager
	IOBuffer(input) |> tokenize |> VNNLib.Parser.TokenManager
end