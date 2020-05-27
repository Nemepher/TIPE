## En réalité, on utilise la corrélation croisé pour éviter d'avoir à rotater l'image ou le filtre.

class Conv:
    def __init__( size,filter_size, padding=0 ): #0 -> valid padding
        self.filter_size = filter_size
        self.padding = padding

