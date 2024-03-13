import math
class ObjectTracker:
    def __init__(self):
        # Memorizza le posizioni centrali degli oggetti
        self.center_points = {}
        # Conta gli ID degli oggetti
        self.id_count = 0

    def update(self, objects_rect):
        # Bounding box degli oggetti e relativi ID
        objects_bbs_ids = []

        # Aggiorna i bounding box degli oggetti
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Controlla se l'oggetto è stato rilevato precedentemente
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            # Se l'oggetto è nuovo, assegna un nuovo ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Rimuove gli oggetti non utilizzati dal dizionario
        self._cleanup_unused_objects(objects_bbs_ids)

        return objects_bbs_ids

    def _cleanup_unused_objects(self, objects_bbs_ids):
        # Rimuove gli oggetti non utilizzati dal dizionario
        used_object_ids = set(obj_bb_id[4] for obj_bb_id in objects_bbs_ids)
        unused_object_ids = set(self.center_points.keys()) - used_object_ids
        for obj_id in unused_object_ids:
            del self.center_points[obj_id]
