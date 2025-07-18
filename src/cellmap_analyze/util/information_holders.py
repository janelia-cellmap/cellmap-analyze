import numpy as np


class ContactingOrganelleInformation:
    def __init__(self, id_to_surface_area_dict={}):
        self.id_to_surface_area_dict = id_to_surface_area_dict

    @staticmethod
    def combine_id_to_surface_area_dicts(dict1, dict2):
        # make dict1 the larger dict
        if len(dict1) < len(dict2):
            dict1, dict2 = dict2, dict1

        dict1 = dict1.copy()
        for id, surface_area in dict2.items():
            dict1[id] = dict1.get(id, 0) + surface_area
        return dict1

    def __add__(self, other: "ContactingOrganelleInformation"):
        coi = ContactingOrganelleInformation()
        coi.id_to_surface_area_dict = (
            ContactingOrganelleInformation.combine_id_to_surface_area_dicts(
                self.id_to_surface_area_dict, other.id_to_surface_area_dict
            )
        )
        return coi

    def __eq__(self, other: "ContactingOrganelleInformation") -> bool:
        return self.id_to_surface_area_dict == other.id_to_surface_area_dict


class ObjectInformation:
    def __init__(
        self,
        counts: int = 0,
        volume: float = 0,
        surface_area: float = 0,
        com: np.ndarray = np.array([0, 0, 0]),
        sum_r2: float = 0,
        bounding_box: list = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf],
        **kwargs,
    ):
        self.counts = counts
        self.volume = volume
        self.surface_area = surface_area
        self.bounding_box = bounding_box
        self.com = com
        self.sum_r2 = sum_r2
        if counts > 0:
            self.radius_of_gyration = np.sqrt(
                self.sum_r2 / counts - np.dot(self.com, self.com)
            )
        else:
            self.radius_of_gyration = np.nan
        self.is_contact_site = False
        if (
            "id_to_surface_area_dict_1" in kwargs.keys()
            or "id_to_surface_area_dict_2" in kwargs.keys()
        ):
            if not (
                "id_to_surface_area_dict_1" in kwargs.keys()
                and "id_to_surface_area_dict_2" in kwargs.keys()
            ):
                raise ValueError(
                    "Must provide both id_to_surface_area_dict_1 and id_to_surface_area_dict_2 if doing contact site analysis"
                )

            self.contacting_organelle_information_1 = ContactingOrganelleInformation(
                kwargs["id_to_surface_area_dict_1"]
            )
            self.contacting_organelle_information_2 = ContactingOrganelleInformation(
                kwargs["id_to_surface_area_dict_2"]
            )
            self.is_contact_site = True

    def __add__(self, other: "ObjectInformation"):
        oi = ObjectInformation()
        oi.counts = self.counts + other.counts
        oi.com = ((self.com * self.volume) + (other.com * other.volume)) / (
            self.volume + other.volume
        )
        oi.volume = self.volume + other.volume
        oi.surface_area = self.surface_area + other.surface_area
        oi.sum_r2 = self.sum_r2 + other.sum_r2
        oi.radius_of_gyration = np.sqrt(oi.sum_r2 / oi.counts - np.dot(oi.com, oi.com))
        if self.is_contact_site != other.is_contact_site:
            raise ValueError(
                "Cannot add ObjectInformation objects with different is_contact_site values"
            )

        if self.is_contact_site:
            oi.is_contact_site = True
            oi.contacting_organelle_information_1 = (
                self.contacting_organelle_information_1
                + other.contacting_organelle_information_1
            )

            oi.contacting_organelle_information_2 = (
                self.contacting_organelle_information_2
                + other.contacting_organelle_information_2
            )

        ndim = len(self.com)
        new_bounding_box = [
            min(self.bounding_box[d], other.bounding_box[d]) for d in range(ndim)
        ]
        new_bounding_box += [
            max(self.bounding_box[d + ndim], other.bounding_box[d + ndim])
            for d in range(ndim)
        ]
        oi.bounding_box = new_bounding_box
        return oi

    def __eq__(self, other: "ObjectInformation") -> bool:
        is_equal = (
            self.volume == other.volume
            and self.surface_area == other.surface_area
            and np.allclose(self.com, other.com, rtol=1e-13, atol=1e-13)
            and np.allclose(self.sum_r2, other.sum_r2, rtol=1e-13, atol=1e-13)
            and np.allclose(
                self.radius_of_gyration,
                other.radius_of_gyration,
                rtol=1e-13,
                atol=1e-13,
            )
            and self.bounding_box == other.bounding_box
            and self.is_contact_site == other.is_contact_site
        )
        if not self.is_contact_site:
            return is_equal
        is_equal &= (
            self.contacting_organelle_information_1
            == other.contacting_organelle_information_1
            and self.contacting_organelle_information_2
            == other.contacting_organelle_information_2
        )
        return is_equal
