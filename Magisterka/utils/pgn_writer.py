import chess.pgn

def save_pgn(game, path, white_name, black_name, extra_headers=None):
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    if extra_headers:
        for k, v in extra_headers.items():
            game.headers[k] = str(v)

    with open(path, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
