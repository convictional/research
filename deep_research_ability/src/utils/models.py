import base64
import binascii
from typing import (
    Any,
    Callable,
    Generic,
    Self,
    Type,
    TypeVar,
)
from urllib.parse import ParseResult, urljoin, urlparse, urlunparse
from uuid import UUID
from uuid import uuid4 as generate_uuid

from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema
from tortoise import BaseDBAsyncClient, Tortoise, fields
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise.exceptions import DoesNotExist, ParamsError
from tortoise.fields.data import T
from tortoise.manager import Manager
from tortoise.models import Model

from ..settings import settings

RecordModelType = TypeVar("RecordModelType", bound="RecordModel")

class ClassProperty(Generic[T]):
    def __init__(self, fget: Callable[[Type[Any]], T]) -> None:
        self.fget = fget

    def __get__(self, instance: Any, owner: Type[Any]) -> T:
        return self.fget(owner)


def classproperty(fget: Callable[[Type[Any]], T]) -> ClassProperty[T]:
    return ClassProperty(fget)


class RecordModel(Model):
    """Simplified RecordModel for experiments without observers/watchers."""
    id = fields.UUIDField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    unscoped: Manager = Manager(Self)

    class Meta:
        abstract = True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.id}>"

    @classproperty  # type: ignore
    def record_type(self) -> str:
        return self.__name__  # type: ignore

    @classmethod
    async def get_or_init(
        cls, defaults: dict | None = None, using_db: BaseDBAsyncClient | None = None, **kwargs: Any
    ) -> Self:
        if not defaults:
            defaults = {}

        db = using_db or cls._choose_db(True)
        try:
            return await cls.filter(**kwargs).using_db(db).get()
        except DoesNotExist:
            for key in defaults.keys() & kwargs.keys():
                if (default_value := defaults[key]) != (query_value := kwargs[key]):
                    raise ParamsError(f"Conflict value with {key=}: {default_value=} vs {query_value=}")
            merged_defaults = {**kwargs, **defaults}
            return cls(**merged_defaults)

    @classmethod
    async def get_or_create(
        cls, defaults: dict | None = None, using_db: BaseDBAsyncClient | None = None, **kwargs: Any
    ) -> tuple[Self, bool]:
        """Get or create a record."""
        instance = await cls.get_or_init(defaults, using_db, **kwargs)
        was_new = not instance._saved_in_db
        if not instance._saved_in_db:
            await instance.save(using_db=using_db)
        return instance, was_new

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type[BaseModel], handler: GetCoreSchemaHandler) -> CoreSchema:
        return pydantic_model_creator(cls).__get_pydantic_core_schema__(source_type, handler)

    @property
    def global_id(self):
        return GlobalID.from_record(self)


class InvalidGlobalIDParamError(Exception):
    pass


class GlobalID(BaseModel):
    """A GlobalID is a unique identifier for a record in the application.
    It is composed of the record type and the record ID.

    Example GlobalID: "gid://decide/DecisionProcess/123e4567-e89b-12d3-a456-426614174000"
    """

    scheme: str = Field(description="URL scheme", default="gid")
    netloc: str = Field(description="Network location", default="decide")
    path: str = Field(description="Path of the record")
    params: str = Field(description="Parameters", default="")
    query: str = Field(description="Query string", default="")
    fragment: str = Field(description="Fragment", default="")

    def __str__(self):
        components = {
            "scheme": self.scheme,
            "netloc": self.netloc,
            "path": self.path,
            "params": self.params,
            "query": self.query,
            "fragment": self.fragment,
        }
        return urlunparse(components.values())

    def __eq__(self, other):
        if other is None:
            return False

        if isinstance(other, str):
            try:
                other = GlobalID.parse(other)
                if not hasattr(other, "app_name"):
                    return False
            except ValueError:
                return False

        return (
            self.app_name == other.app_name
            and self.record_type == other.record_type
            and self.record_id == other.record_id
        )

    def __hash__(self):
        return hash((self.app_name, self.record_type, self.record_id))

    @classmethod
    def parse(cls, gid: str) -> Self:
        components: ParseResult = urlparse(gid)
        return cls(
            scheme=components.scheme,
            netloc=components.netloc,
            path=components.path,
            params=components.params,
            query=components.query,
            fragment=components.fragment,
        )

    @classmethod
    def create(cls, record_type: str, record_id: UUID, app_name: str = "decide"):
        components = {
            "scheme": "gid",
            "netloc": app_name,
            "path": f"/{record_type}/{record_id}",
            "params": "",
            "query": "",
            "fragment": "",
        }
        return cls(**components)

    @classmethod
    def from_record(cls, record: RecordModel, app_name: str = "decide"):
        components = {
            "scheme": "gid",
            "netloc": app_name,
            "path": f"/{record.record_type}/{record.id}",
            "params": "",
            "query": "",
            "fragment": "",
        }
        return cls(**components)

    @classmethod
    def from_param(cls, param: str) -> Self:
        try:
            decoded = base64.b64decode(param.encode()).decode()
            return cls.parse(decoded)
        except binascii.Error as e:
            raise InvalidGlobalIDParamError(f"Invalid base64 encoding: {e}")
        except UnicodeDecodeError as e:
            raise InvalidGlobalIDParamError(f"Unable to decode parameter: {e}")
        except ValueError as e:
            raise InvalidGlobalIDParamError(f"Unable to parse decoded value: {e}")

    @property
    def is_internal(self):
        return self.scheme == "gid"

    @property
    def record_type(self):
        if self.is_internal:
            return self.path.strip("/").split("/")[0]
        return None

    @property
    def record_id(self):
        if self.is_internal:
            id_str = self.path.strip("/").split("/")[1]
            return UUID(id_str)
        return None

    @property
    def app_name(self):
        if self.is_internal:
            return self.netloc
        return None

    @property
    def to_param(self):
        return base64.b64encode(str(self).encode()).decode()

    @property
    def to_url(self):
        if self.is_internal:
            base_url = str(settings.base_url)
            return urljoin(base_url, f"/gid/{self.to_param}")
        else:
            return str(self)

    async def get_or_none(self, using_db: BaseDBAsyncClient | None = None) -> RecordModel | None:
        model = Tortoise.apps[self.app_name][self.record_type]
        if not issubclass(model, RecordModel):
            raise ValueError("Model must be a subclass of RecordModel")

        return await model.get_or_none(id=self.record_id, using_db=using_db)

    async def get(self, using_db: BaseDBAsyncClient | None = None) -> RecordModel:
        model = await self.get_or_none(using_db)
        if not model:
            raise DoesNotExist(f"No record found for {self}")

        return model
