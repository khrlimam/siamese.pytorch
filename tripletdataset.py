from torch.utils.data import Dataset
import pathlib


class TripletDataset(Dataset):
    def __init__(self, path, ext, transforms):
        super(TripletDataset, self).__init__()
        self.path = path
        self.ext = ext
        self.transforms = transforms
        path = pathlib.Path(path)
        self.files = sorted(path.glob(f"*/*.{ext}"))
        self.grouped_by_class = self._group_by_class(self.files)
        self.apns = self.generate_anchor_positive_negative(self.files)

    def _group_by_class(self, files):
        classes = set(list(map(lambda x: x.parent.name, files)))
        return {class_: list(filter(lambda x: x.parent.name == class_, files)) for class_ in classes}

    def generate_anchor_positive_negative(self, files):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
