import time
from tqdm import tqdm


# Função para monitorar o progresso das threads
def monitor_progress(threads):
    with tqdm(total=len(threads)) as pbar:
        for thread in threads:
            while thread.is_alive():
                pbar.update(0)  # Atualiza a barra de progresso
                time.sleep(1)  # Aguarda um segundo
            pbar.update(1)  # Incrementa o progresso quando a thread termina