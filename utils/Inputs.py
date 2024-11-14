class Input:
    def __init__(self, name, description, field_type, options):
        self.name = name
        self.type = field_type
        self.description = description
        self.options = options


inputs = [
    Input('LotArea', 'Lot size in square feet', 'numeric', []),
    Input('Neighborhood', 'Physical locations within Ames city limits', 'select',
          {'Blmngtn': 'Bloomington Heights', 'Blueste': 'Bluestem', 'BrDale': 'Briardale', 'BrkSide': 'Brookside',
           'ClearCr': 'Clear Creek', 'CollgCr': 'College Creek', 'Crawfor': 'Crawford', 'Edwards': 'Edwards',
           'Gilbert': 'Gilbert', 'IDOTRR': 'Iowa DOT and Rail Road', 'MeadowV': 'Meadow Village', 'Mitchel': 'Mitchell',
           'Names': 'North Ames', 'NoRidge': 'Northridge', 'NPkVill': 'Northpark Villa',
           'NridgHt': 'Northridge Heights', 'NWAmes': 'Northwest Ames', 'OldTown': 'Old Town',
           'SWISU': 'South & West of Iowa State University', 'Sawyer': 'Sawyer', 'SawyerW': 'Sawyer West',
           'Somerst': 'Somerset', 'StoneBr': 'Stone Brook', 'Timber': 'Timberland', 'Veenker': 'Veenker'}),
    Input('OverallQual', 'Rates the overall material and finish of the house', 'select',
          {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
           4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor'}),
    Input('OverallCond', 'Rates the overall condition of the house', 'select',
          {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
           4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor'}),
    Input('YearBuilt', 'Original construction date', 'numeric', []),
    Input('YearRemodAdd', 'Remodel date (same as construction date if no remodeling or additions)', 'numeric', []),
    Input('BsmtQual', 'Evaluates the height of the basement', 'select',
          {'Ex': 'Excellent (100+ inches)', 'Gd': 'Good (90-99 inches)', 'TA': 'Typical (80-89 inches)',
           'Fa': 'Fair (70-79 inches)', 'Po': 'Poor (<70 inches)', 'NA': 'No Basement'}),
    Input('BsmtExposure', 'Refers to walkout or garden level walls', 'select',
          {'Gd': 'Good Exposure', 'Av': 'Average Exposure', 'Mn': 'Minimum Exposure', 'No': 'No Exposure',
           'NA': 'No Basement'}),
    Input('BsmtFinSF1', 'Type 1 finished square feet', 'numeric', []),
    Input('TotalBsmtSF', 'Total square feet of basement area', 'numeric', []),
    Input('FirstFlrSF', 'First Floor square feet', 'numeric', []),
    Input('SecondFlrSF', 'Second floor square feet', 'numeric', []),
    Input('GrLivArea', 'Above grade (ground) living area square feet', 'numeric', []),
    Input('FullBath', 'Full bathrooms above grade', 'numeric', []),
    Input('KitchenQual', 'Kitchen quality', 'select',
          {'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'}),
    Input('Fireplaces', 'Number of fireplaces', 'numeric', []),
    Input('GarageCars', 'Size of garage in car capacity', 'numeric', []),
    Input('GarageArea', 'Size of garage in square feet', 'numeric', []),
    Input('OpenPorchSF', 'Open porch area in square feet', 'numeric', []),
    # Input('SalePrice', 'Sale price of the property', 'numeric', []),
]
