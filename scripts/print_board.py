def print_board(board:list = [0, 1, 2, 3, 4, 5, 6, 7, 8]):
    """
    Função para desenhar o tabuleiro com os dados enviados

    Parâmetros
    ----------
    board: list
        Lista de valores do tabuleiro
    """


    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("___|___|___")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("___|___|___")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("   |   |")