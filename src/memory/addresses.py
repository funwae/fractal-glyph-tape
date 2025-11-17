"""Logical addressing schema and utilities for Fractal Glyph Memory Service."""

import re
from typing import List

from .models import FractalAddress


# Address string format: "world/region#tri_path@dDEPTHtTIME"
# Example: "earthcloud/hayden-agent#573@d2t17"
# tri_path is encoded as a comma-separated list in full form: #1,2,3

ADDRESS_PATTERN = re.compile(
    r"^(?P<world>[^/]+)/(?P<region>[^#]+)#(?P<tri_path>[\d,]+)@d(?P<depth>\d+)t(?P<time_slice>\d+)$"
)


def address_to_str(addr: FractalAddress) -> str:
    """Serialize a FractalAddress to string format.

    Args:
        addr: FractalAddress to serialize

    Returns:
        String representation of the address

    Example:
        >>> addr = FractalAddress("default", "agent:hayden", [1, 2, 3], 2, 17)
        >>> address_to_str(addr)
        'default/agent:hayden#1,2,3@d2t17'
    """
    tri_path_str = ",".join(str(x) for x in addr.tri_path)
    if not tri_path_str:
        tri_path_str = "0"  # Default for empty path
    return f"{addr.world}/{addr.region}#{tri_path_str}@d{addr.depth}t{addr.time_slice}"


def address_from_str(s: str) -> FractalAddress:
    """Deserialize a string to FractalAddress.

    Args:
        s: String representation of address

    Returns:
        FractalAddress object

    Raises:
        ValueError: If string format is invalid

    Example:
        >>> addr = address_from_str("default/agent:hayden#1,2,3@d2t17")
        >>> addr.world
        'default'
        >>> addr.tri_path
        [1, 2, 3]
    """
    match = ADDRESS_PATTERN.match(s)
    if not match:
        raise ValueError(f"Invalid address format: {s}")

    groups = match.groupdict()

    # Parse tri_path
    tri_path_str = groups["tri_path"]
    if tri_path_str == "0":
        tri_path = []
    else:
        try:
            tri_path = [int(x) for x in tri_path_str.split(",")]
        except ValueError:
            raise ValueError(f"Invalid tri_path in address: {tri_path_str}")

    return FractalAddress(
        world=groups["world"],
        region=groups["region"],
        tri_path=tri_path,
        depth=int(groups["depth"]),
        time_slice=int(groups["time_slice"]),
    )


def validate_address_str(s: str) -> bool:
    """Check if a string is a valid address format.

    Args:
        s: String to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> validate_address_str("default/agent:hayden#1,2,3@d2t17")
        True
        >>> validate_address_str("invalid")
        False
    """
    return ADDRESS_PATTERN.match(s) is not None


def create_address(
    world: str,
    region: str,
    tri_path: List[int],
    depth: int = 0,
    time_slice: int = 0,
) -> FractalAddress:
    """Create a new FractalAddress with validation.

    Args:
        world: World/namespace
        region: Region/topic
        tri_path: Triangular fractal path
        depth: Depth level (default: 0)
        time_slice: Time slice index (default: 0)

    Returns:
        FractalAddress object

    Raises:
        ValueError: If any parameters are invalid

    Example:
        >>> addr = create_address("default", "agent:hayden", [1, 2, 3], depth=2, time_slice=5)
        >>> address_to_str(addr)
        'default/agent:hayden#1,2,3@d2t5'
    """
    return FractalAddress(
        world=world,
        region=region,
        tri_path=tri_path if tri_path else [],
        depth=depth,
        time_slice=time_slice,
    )


def parse_region_from_actor(actor_id: str, region_prefix: str = "agent") -> str:
    """Generate a default region identifier from an actor ID.

    Args:
        actor_id: Actor/user identifier
        region_prefix: Prefix for the region (default: "agent")

    Returns:
        Region string

    Example:
        >>> parse_region_from_actor("hayden")
        'agent:hayden'
        >>> parse_region_from_actor("user123", "user")
        'user:user123'
    """
    return f"{region_prefix}:{actor_id}"


def addresses_equal(addr1: FractalAddress, addr2: FractalAddress) -> bool:
    """Check if two addresses are equal.

    Args:
        addr1: First address
        addr2: Second address

    Returns:
        True if addresses are equal, False otherwise

    Example:
        >>> addr1 = create_address("default", "agent:hayden", [1, 2, 3])
        >>> addr2 = create_address("default", "agent:hayden", [1, 2, 3])
        >>> addresses_equal(addr1, addr2)
        True
    """
    return (
        addr1.world == addr2.world
        and addr1.region == addr2.region
        and addr1.tri_path == addr2.tri_path
        and addr1.depth == addr2.depth
        and addr1.time_slice == addr2.time_slice
    )


def get_address_depth_level(addr: FractalAddress) -> int:
    """Get the depth level of an address.

    Args:
        addr: Address to query

    Returns:
        Depth level (0 = summary, higher = more detail)

    Example:
        >>> addr = create_address("default", "agent:hayden", [1, 2, 3], depth=2)
        >>> get_address_depth_level(addr)
        2
    """
    return addr.depth


def get_address_region(addr: FractalAddress) -> str:
    """Get the region of an address.

    Args:
        addr: Address to query

    Returns:
        Region string

    Example:
        >>> addr = create_address("default", "agent:hayden", [1, 2, 3])
        >>> get_address_region(addr)
        'agent:hayden'
    """
    return addr.region


def increment_time_slice(addr: FractalAddress) -> FractalAddress:
    """Create a new address with incremented time slice.

    Args:
        addr: Original address

    Returns:
        New address with time_slice + 1

    Example:
        >>> addr1 = create_address("default", "agent:hayden", [1, 2, 3], time_slice=5)
        >>> addr2 = increment_time_slice(addr1)
        >>> addr2.time_slice
        6
    """
    return FractalAddress(
        world=addr.world,
        region=addr.region,
        tri_path=addr.tri_path,
        depth=addr.depth,
        time_slice=addr.time_slice + 1,
    )


def with_depth(addr: FractalAddress, depth: int) -> FractalAddress:
    """Create a new address with a different depth.

    Args:
        addr: Original address
        depth: New depth level

    Returns:
        New address with specified depth

    Example:
        >>> addr1 = create_address("default", "agent:hayden", [1, 2, 3], depth=0)
        >>> addr2 = with_depth(addr1, 2)
        >>> addr2.depth
        2
    """
    return FractalAddress(
        world=addr.world,
        region=addr.region,
        tri_path=addr.tri_path,
        depth=depth,
        time_slice=addr.time_slice,
    )
