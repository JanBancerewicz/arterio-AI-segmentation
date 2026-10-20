const dateFormatter = new Intl.DateTimeFormat('pl-PL', {
    dateStyle: 'short',
    timeStyle: 'short',
});

export const formatDate = (isoDate: string) => dateFormatter.format(new Date(isoDate));
