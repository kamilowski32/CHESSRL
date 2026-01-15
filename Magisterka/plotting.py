import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    # Ustawienia figury
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Style
    box_style = dict(boxstyle="round,pad=0.3", ec="black", lw=1.5)
    arrow_props = dict(arrowstyle="-|>", color="#333333", lw=1.5, shrinkA=5, shrinkB=5)
    double_arrow = dict(arrowstyle="<|-|>", color="#333333", lw=1.5, shrinkA=5, shrinkB=5)

    # --- Pozycje Bloków ---
    # Trener w centrum (jako koordynator)
    trainer_pos = (6, 4.5)
    # Agent po lewej
    agent_pos = (2.5, 4.5)
    # Środowisko po prawej
    env_pos = (9.5, 4.5)
    # Pula agentów na dole
    pool_pos = (6, 1.5)

    # --- Funkcja rysująca blok ---
    def draw_node(pos, title, subtitle, color, width=2.5, height=1.2):
        rect = patches.FancyBboxPatch(
            (pos[0] - width/2, pos[1] - height/2), width, height,
            boxstyle="round,pad=0.2",
            linewidth=1.5, edgecolor='black', facecolor=color,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(pos[0], pos[1] + 0.2, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(pos[0], pos[1] - 0.25, subtitle, ha='center', va='center', fontsize=9, style='italic')

    # --- Rysowanie Modułów ---
    
    # 3. Trener (Centrum)
    draw_node(trainer_pos, "3. Trener", "Zarządzanie pętlą\ni aktualizacja wag", "#FFF9C4") # Żółty

    # 2. Agent (Lewa)
    draw_node(agent_pos, "2. Agent", "Sieć neuronowa\n(Polityka / Wartość)", "#BBDEFB") # Niebieski

    # 1. Środowisko (Prawa)
    draw_node(env_pos, "1. Środowisko", "Logika gry\ni walidacja ruchów", "#C8E6C9") # Zielony

    # 4. Pula Agentów (Dół)
    # Rysujemy jako "bazę danych" lub stos
    draw_node(pool_pos, "4. Pula Agentów", "Wersje historyczne\ni referencyjne", "#E1BEE7") # Fioletowy

    # --- Rysowanie Relacji (Strzałki) ---

    # A. Trener <-> Agent
    # Strzałka w lewo (Pobranie decyzji)
    ax.annotate("", xy=(agent_pos[0]+1.25, agent_pos[1]+0.2), xytext=(trainer_pos[0]-1.25, trainer_pos[1]+0.2), arrowprops=arrow_props)
    ax.text(4.25, 4.8, "Stan", ha='center', va='bottom', fontsize=9)
    
    # Strzałka w prawo (Decyzja / Update)
    ax.annotate("", xy=(trainer_pos[0]-1.25, trainer_pos[1]-0.2), xytext=(agent_pos[0]+1.25, agent_pos[1]-0.2), arrowprops=arrow_props)
    ax.text(4.25, 4.2, "Akcja / Logity", ha='center', va='top', fontsize=9)
    
    # Dodatkowa strzałka aktualizacji wag (pod spodem)
    ax.annotate("", xy=(agent_pos[0]+1.25, agent_pos[1]-0.6), xytext=(trainer_pos[0]-1.25, trainer_pos[1]-0.6), 
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2, shrinkA=5, shrinkB=5, ls="--"))
    ax.text(4.25, 3.7, "Aktualizacja wag (Backprop)", ha='center', va='center', fontsize=8, color="red")

    # B. Trener <-> Środowisko
    # Strzałka w prawo (Wykonanie ruchu)
    ax.annotate("", xy=(env_pos[0]-1.25, env_pos[1]+0.2), xytext=(trainer_pos[0]+1.25, trainer_pos[1]+0.2), arrowprops=arrow_props)
    ax.text(7.75, 4.8, "Wykonaj ruch", ha='center', va='bottom', fontsize=9)

    # Strzałka w lewo (Nowy stan)
    ax.annotate("", xy=(trainer_pos[0]+1.25, trainer_pos[1]-0.2), xytext=(env_pos[0]-1.25, env_pos[1]-0.2), arrowprops=arrow_props)
    ax.text(7.75, 4.2, "Nowy Stan, Nagroda", ha='center', va='top', fontsize=9)

    # C. Trener <-> Pula Agentów
    # Zapis
    ax.annotate("", xy=(pool_pos[0]-0.5, pool_pos[1]+0.6), xytext=(trainer_pos[0]-0.5, trainer_pos[1]-0.6), arrowprops=arrow_props)
    ax.text(5.3, 3.0, "Zapis modelu", ha='right', va='center', fontsize=9, rotation=0)

    # Odczyt (np. do ewaluacji)
    ax.annotate("", xy=(trainer_pos[0]+0.5, trainer_pos[1]-0.6), xytext=(pool_pos[0]+0.5, pool_pos[1]+0.6), arrowprops=arrow_props)
    ax.text(6.7, 3.0, "Ładowanie przeciwnika", ha='left', va='center', fontsize=9, rotation=0)

    # Tytuł
    plt.title("Rysunek 3.3: Schemat architektury systemu i przepływ danych", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_architecture_diagram()