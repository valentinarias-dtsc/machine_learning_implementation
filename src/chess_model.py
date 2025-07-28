import numpy as np
import chess
import chess.svg
import random
from wand.image import Image as WandImage
import io
from PIL import Image

# Piece-Square Tables for each piece type (simplified example)
PIECE_SQUARE_TABLES = {
    chess.KING : [-65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14],

chess.QUEEN : [-28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50],

chess.ROOK : [32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26],

chess.BISHOP : [-29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21],

chess.KNIGHT : [-167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23],

chess.PAWN : [ 0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0]
}

def piece_square_score(board, piece_type, all_pieces):
    """Optimizado para usar la información de piezas precomputada."""
    positions_white = all_pieces[piece_type][chess.WHITE].mirror()
    positions_black = all_pieces[piece_type][chess.BLACK]

    pst = np.array(PIECE_SQUARE_TABLES[piece_type])
    score_white = sum(pst[pos] for pos in positions_white)
    score_black = sum(pst[pos] for pos in positions_black)
    return score_white - score_black

def calculate_features(board):
    """Optimizado para reducir llamadas a board.pieces."""
    features = {}

    # Recuperar todas las piezas de una vez
    all_pieces = {
        piece: {
            chess.WHITE: board.pieces(piece, chess.WHITE),
            chess.BLACK: board.pieces(piece, chess.BLACK)
        }
        for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    }

    # Diferencias de piezas
    for piece, name in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN],
                           ["pawn", "knight", "bishop", "rook", "queen"]):
        wp = len(all_pieces[piece][chess.WHITE])
        bp = len(all_pieces[piece][chess.BLACK])
        features[f"diff_{name}"] = wp - bp

    # Piece-square table
    for piece, name in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN],
                           ["pawn", "knight", "bishop", "rook", "queen"]):
        features[f"pst_{name}"] = piece_square_score(board, piece, all_pieces)

    return features

def evaluate(board, coef):

    if board.is_game_over():
        # Evalúa el resultado final de la partida
        if board.result() == "1-0":
            return 1000000000000  # Victoria para blancas
        elif board.result() == "0-1":
            return -1000000000000  # Victoria para negras
        else:
            return 0  # Empate

    # Calcular características solo si no están en la cache
    features = calculate_features(board)
    evaluation = np.dot(np.array(list(features.values())), coef[1:].T)
    return evaluation

def alpha_beta(board, depth, alpha, beta, maximizing_player, coef):
    if depth == 0 :
        return evaluate(board, coef), []

    best_move = None
    best_variation = []

    # Ordenar movimientos antes de iterar sobre ellos
    legal_moves = sorted(board.legal_moves, key=lambda move: score_move(board, move), reverse=True)

    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            board.push(move)
            eval, variation = alpha_beta(
                board, 
                depth - 1 , 
                alpha, 
                beta, 
                False,
                coef
            )
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move
                best_variation = [move] + variation

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_variation
    else:
        min_eval = float("inf")
        for move in legal_moves:
            

            board.push(move)
            eval, variation = alpha_beta(
                board, 
                depth - 1 , 
                alpha, 
                beta, 
                True,
                coef
            )
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move
                best_variation = [move] + variation

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return min_eval, best_variation

def score_move(board, move):
    """
    Función para puntuar un movimiento basado en MVV-LVA, promociones y jaques.
    Los movimientos con puntuaciones más altas serán evaluados primero.
    """
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        # MVV-LVA: Mayor valor de la víctima y menor del atacante
        return (victim.piece_type * 10 - attacker.piece_type) if victim and attacker else 0
    elif move.promotion:
        return 9  # Priorizar promociones
    elif board.gives_check(move):
        return 5  # Priorizar jaques
    return 0  # Movimientos no especiales tienen prioridad baja

def best_move(board, depth, coef):
    best_eval = float('-inf') if board.turn else float('inf')
    best_move = None
    mejores = []
    
    for move in sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True):
        board.push(move)
        eval,variante = alpha_beta(board, depth - 1, float('-inf'), float('inf'), board.turn, coef)
        board.pop()
        if (board.turn and eval > best_eval) or (not board.turn and eval < best_eval):
            mejores.append(move)
            best_eval = eval
            best_move = move
        if (board.turn and eval == best_eval) or (not board.turn and eval == best_eval):
            mejores.append(move)
        
        #print("La evaluación del movimiento {} es {}.".format(move,eval))
        #print("La variante crítica después de {} es: {}".format(move,[str(i) for i in variante]))
        #print("")
    return random.choice(mejores), best_eval
def run(path, coef):
    # Crear un tablero inicial. Gambito Marshall. 
    board = chess.Board(fen='r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - d6 0 9')

    # Lista para almacenar los frames del GIF
    frames = []

    for i in range(70):
            # Encontrar la mejor jugada con profundidad 2
            move,eval = best_move(board, 2, coef)
            print(f"Best move: {move}")
            print("Evaluation: {}".format(eval))

            # Realizar la jugada en el tablero
            board.push(move)

            # Generar el SVG del tablero
            svg_image = chess.svg.board(board, size=350)

            # Convertir el SVG a PNG usando Wand
            with WandImage(blob=svg_image.encode("utf-8"), format="svg") as img:
                img.format = "png"
                png_data = io.BytesIO(img.make_blob("png")) 
                frame = Image.open(png_data).convert("RGBA")
                frames.append(frame) 

    # Guardar las imágenes como un GIF animado 
    frames[0].save(str(path), save_all=True, append_images=frames[1:], duration=1000, loop=0)

    print(f"GIF generated: {path}")