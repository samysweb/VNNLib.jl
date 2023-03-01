function generate_token(text :: String, kind::Tokens.Kind)
    return Tokens.Token(kind,(1,1),(1,2),1,2,text)
end