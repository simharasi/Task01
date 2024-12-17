def find_min_string_factor(X, Y, S, R):
    n = len(X)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  
    
    for i in range(n):
        if dp[i] == float('inf'):
            continue
        
        
        for length in range(1, n - i + 1):
            subX = X[i:i + length]
            
           
            if subX in Y:
                dp[i + length] = min(dp[i + length], dp[i] + S)
            
            revSubX = subX[::-1]
            if revSubX in Y:
                dp[i + length] = min(dp[i + length], dp[i] + R)
    
    
    return dp[n] if dp[n] != float('inf') else "Impossible"


X = input().strip()
Y = input().strip()
S, R = map(int, input().strip().split())


result = find_min_string_factor(X, Y, S, R)
print(result)