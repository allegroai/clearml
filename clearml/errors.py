class UsageError(RuntimeError):
    """ An exception raised for illegal usage of clearml objects"""
    pass


class ArtifactUriDeleteError(ValueError):
    def __init__(self, artifact, uri, remaining_uris):
        super(ArtifactUriDeleteError, self).__init__("Failed deleting artifact {}: file {}".format(artifact, uri))
        self.artifact = artifact
        self.uri = uri
        self._remaining_uris = remaining_uris

    @property
    def remaining_uris(self):
        """ Remaining URIs to delete. Deletion of these URIs was aborted due to the error. """
        return self._remaining_uris
