from agent import retrieve,answerer


if __name__ == "__main__":
    query = "What is the capital of France"
    answer = answerer(query)
    print(answer)