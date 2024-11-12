from typing import Literal, Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, validator


class ContourConfig(BaseModel):
    """
    The contour is a rectangle, going around the eigenvalues of the sample covariance matrix. To make sure
    the full limit spectral distribution is captured, the contour is extended by a slack in the real. The height
    of the rectangle is given by the imag_height parameter.
    """

    real_slack: NonNegativeFloat  # slack to add around the estimated eigenvalues
    type_: Literal["circle"]  # only circle is implemented currently
    imag_height: Optional[PositiveFloat] = (
        None  # height of the contour in the imaginary axis, not defined in case of circle
    )

    # add a validator that checks that real_slack is None if type_ is circle
    @validator("imag_height")
    def check_imag_height(cls, v, values):
        if values["type_"] == "circle":
            if v is not None:
                raise ValueError("imag_height should be None if type_ is circle")
        return v


class ContourPairConfig(BaseModel):
    contours: tuple[ContourConfig, ContourConfig]

    @validator("contours")
    def validate_contour_pair_config(cls, v):
        """
        Validate that ct1 is strictly inside ct2 (or the other
        way around)
        """
        ct1, ct2 = v

        if (ct1.imag_height is not None) and (ct2.imag_height is not None):
            is_ct1_inside_ct2 = (
                ct1.imag_height < ct2.imag_height and ct1.real_slack < ct2.real_slack
            )
            is_ct2_inside_ct1 = (
                ct2.imag_height < ct1.imag_height and ct2.real_slack < ct1.real_slack
            )
        else:
            is_ct1_inside_ct2 = ct1.real_slack < ct2.real_slack
            is_ct2_inside_ct1 = ct2.real_slack < ct1.real_slack

        if is_ct1_inside_ct2 or is_ct2_inside_ct1:
            return v
        else:
            raise ValueError("One contour must be strictly inside the other")
